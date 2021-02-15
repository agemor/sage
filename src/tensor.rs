use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;
use std::mem;

pub type Tensor = ArrayD<f32>;
pub type TensorView<'a> = ArrayViewD<'a, f32>;

fn fan_in_and_out(shape: &Shape) -> (usize, usize) {
    if shape.ndim() < 2 {
        panic!("cannot computed.. shape too small");
    }

    let num_in_fmaps = shape.dim[1];
    let num_out_fmaps = shape.dim[0];

    let mut receptive_field_size = 1;

    if shape.ndim() > 2 {
        receptive_field_size = shape.dim[2..].iter().fold(1, |a, b| a * (*b));
    }

    let fan_in = num_in_fmaps * receptive_field_size;
    let fan_out = num_out_fmaps * receptive_field_size;

    (fan_in, fan_out)
}

pub fn kaiming_uniform(shape: &Shape, gain: f32) -> Tensor {
    let (fan_in, _) = fan_in_and_out(shape);
    let std = gain * (1.0 / fan_in as f32).sqrt();
    let a = 3.0_f32.sqrt() * std;

    Tensor::random(IxDyn(&shape.dim), Uniform::new(-a, a))
}

pub fn kaiming_normal(shape: &Shape, gain: f32) -> Tensor {
    let (fan_in, _) = fan_in_and_out(shape);
    let std = gain * (1.0 / fan_in as f32).sqrt();
    Tensor::random(IxDyn(&shape.dim), Normal::new(0.0, std).unwrap())
}

pub fn from_vec(shape: Shape, vec: Vec<f32>) -> Tensor {
    Tensor::from_shape_vec(IxDyn(&shape.dim), vec).unwrap()
}

pub fn zeros(shape: Shape) -> Tensor {
    Tensor::zeros(IxDyn(&shape.dim))
}

pub fn ones(shape: &Shape) -> Tensor {
    Tensor::ones(IxDyn(&shape.dim))
}

pub fn mem_size(x: &Tensor) -> usize {
    mem::size_of_val(x) + x.len() * mem::size_of::<f32>()
}

pub fn logsumexp(x: TensorView, axis: usize, keep_dims: bool) -> Tensor {
    let mut shape = x.shape().to_vec();

    if keep_dims {
        shape[axis] = 1;
    } else {
        shape.remove(axis);
    }

    let max = (&x)
        .fold_axis(
            Axis(axis),
            f32::MIN,
            move |&a, &b| if a > b { a } else { b },
        )
        .into_shape(IxDyn(&shape))
        .unwrap();

    let mut y: Tensor = &x - &max;
    y.mapv_inplace(|x| x.exp());

    let mut sum = y.sum_axis(Axis(axis)).into_shape(IxDyn(&shape)).unwrap();
    sum.mapv_inplace(move |a| a.ln());
    sum += &max;
    sum
}

// (*, A, B) (*, B) -> (*, A)
pub fn matvec(a: TensorView, b: TensorView) -> Result<Tensor, ShapeError> {
    let a_dim = a.ndim();
    let b_dim = b.ndim();

    // ensure at least two dims
    if a_dim < 2 || b_dim < 1 {
        return Err(ShapeError::new("not a matrix or vector"));
    }

    // check last two dims are compatible,
    if a.shape()[a_dim - 1] != b.shape()[b_dim - 1] {
        return Err(ShapeError::new("matrix and vector are not compatible"));
    }

    // if a_dim=2, b_dim =2 return matmul
    if a_dim == 2 && b_dim == 1 {
        let a2d = a.into_dimensionality::<Ix2>().unwrap();
        let b1d = b.into_dimensionality::<Ix1>().unwrap();
        let c1d = a2d.dot(&b1d);
        Ok(c1d.into_dyn())
    } else {
        // create a shared shape
        let (a_bat_shape, a_mat_shape) = a.shape().split_at(a_dim - 2);
        let (b_bat_shape, b_mat_shape) = b.shape().split_at(b_dim - 1);

        // shape broadcast
        let c_bat_shape = broadcast(a_bat_shape, b_bat_shape)?;

        let mut a_shape = c_bat_shape.clone();
        let mut b_shape = c_bat_shape.clone();

        a_shape.extend_from_slice(a_mat_shape);
        b_shape.extend_from_slice(b_mat_shape);

        // real broadcast
        let a = a.broadcast(a_shape).unwrap();
        let b = b.broadcast(b_shape).unwrap();

        let axis0_len = c_bat_shape[0];

        let mut c = Vec::<Tensor>::with_capacity(axis0_len);

        for i in 0..axis0_len {
            let a_i: TensorView = a.index_axis(Axis(0), i);
            let b_i: TensorView = b.index_axis(Axis(0), i);
            let c_i = matvec(a_i, b_i).unwrap();
            c.push(c_i);
        }

        let c_view: Vec<TensorView> = c.iter().map(|c| c.view()).collect();

        if let Ok(stacked) = ndarray::stack(Axis(0), &c_view) {
            Ok(stacked)
        } else {
            Err(ShapeError::new("stacking failed"))
        }
    }
}

// (*, A, B) (*, B, C) -> (*, A, C)
pub fn matmul(a: TensorView, b: TensorView) -> Result<Tensor, ShapeError> {
    let a_dim = a.ndim();
    let b_dim = b.ndim();

    // ensure at least two dims
    if a_dim < 2 || b_dim < 2 {
        return Err(ShapeError::new("not a matrix"));
    }

    // check last two dims are compatible,
    if a.shape()[a_dim - 1] != b.shape()[b_dim - 2] {
        return Err(ShapeError::new("matrix not compatible"));
    }

    // if a_dim=2, b_dim =2 return matmul
    if a_dim == 2 && b_dim == 2 {
        let a2d = a.into_dimensionality::<Ix2>().unwrap();
        let b2d = b.into_dimensionality::<Ix2>().unwrap();
        let c2d = a2d.dot(&b2d);
        Ok(c2d.into_dyn())
    } else {
        // create a shared shape
        let (a_bat_shape, a_mat_shape) = a.shape().split_at(a_dim - 2);
        let (b_bat_shape, b_mat_shape) = b.shape().split_at(b_dim - 2);

        // shape broadcast
        let c_bat_shape = broadcast(a_bat_shape, b_bat_shape)?;

        let mut a_shape = c_bat_shape.clone();
        let mut b_shape = c_bat_shape.clone();

        a_shape.extend_from_slice(a_mat_shape);
        b_shape.extend_from_slice(b_mat_shape);

        // real broadcast
        let a = a.broadcast(a_shape).unwrap();
        let b = b.broadcast(b_shape).unwrap();

        let axis0_len = c_bat_shape[0];

        let mut c = Vec::<Tensor>::with_capacity(axis0_len);

        for i in 0..axis0_len {
            let a_i: TensorView = a.index_axis(Axis(0), i);
            let b_i: TensorView = b.index_axis(Axis(0), i);
            let c_i = matmul(a_i, b_i).unwrap();
            c.push(c_i);
        }

        let c_view: Vec<TensorView> = c.iter().map(|c| c.view()).collect();

        if let Ok(stacked) = ndarray::stack(Axis(0), &c_view) {
            Ok(stacked)
        } else {
            Err(ShapeError::new("stacking failed"))
        }
    }
}

// make tensors from data

// make empty tensors

#[derive(Clone, Eq, PartialEq)]
pub struct Shape {
    pub dim: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ShapeError {
    pub msg: String,
}

impl ShapeError {
    pub(crate) fn new(msg: &str) -> Self {
        ShapeError {
            msg: msg.to_string(),
        }
    }
}

impl Shape {
    pub fn new(dim: &[usize]) -> Shape {
        Shape { dim: dim.to_vec() }
    }

    pub fn ndim(&self) -> usize {
        self.dim.len()
    }

    pub fn expand_dim(&mut self, axis: usize) {
        self.dim.insert(axis, 1);
    }

    pub fn broadcast(&self, other: &Shape) -> Result<Shape, ShapeError> {
        let dim = broadcast(&self.dim, &other.dim)?;
        Ok(Shape::new(&dim))
    }
}

pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeError> {
    if a == b {
        Ok(a.to_vec())
    }
    // Do broadcasting
    else {
        let (longer, shorter) = if a.len() > b.len() { (a, b) } else { (b, a) };

        let mut padded = longer[0..(longer.len() - shorter.len())].to_vec();
        padded.extend_from_slice(shorter);

        let mut result = Vec::<usize>::with_capacity(longer.len());

        for (a, b) in longer.iter().zip(padded.iter()) {
            if *a == 1 || *a == *b {
                result.push(*b);
            } else if *b == 1 {
                result.push(*a);
            } else {
                return Err(ShapeError::new("invalid broadcast"));
            }
        }

        Ok(result)
    }
}
