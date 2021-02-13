use ndarray::prelude::*;
use ndarray::{IxDynImpl, OwnedRepr};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::cmp::max;

pub type Tensor = ArrayD<f32>;
pub type TensorView<'a> = ArrayViewD<'a, f32>;

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
            return Err(ShapeError::new("stacking failed"));
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
            return Err(ShapeError::new("stacking failed"));
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
