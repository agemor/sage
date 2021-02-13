use crate::autodiff::{op, Op, Var};
use crate::tensor;
use crate::tensor::{Shape, ShapeError, Tensor};
use ndarray::Axis;
use ndarray_stats::QuantileExt;
use std::ops;
use std::ops::Neg as Neg2;

struct Add;
struct Sub;
struct Neg;
struct Mul;
struct Div;

struct Sum {
    axis: usize,
}

struct SumTo {
    shape: Shape,
}
struct BroadcastTo {
    shape: Shape,
}

struct MatMul;
struct Transpose;

struct ReLU;
struct Binarize {
    threshold: f32,
}

struct Softmax {
    axis: usize,
}
struct SoftmaxCrossEntropy;

impl Op for Add {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];
        x0 + x1
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];
        x0.shape().broadcast(x1.shape())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = sum_to(gy, x0.shape());
        let gx1 = sum_to(gy, x1.shape());

        vec![gx0, gx1]
    }
}

impl Op for Sub {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];
        x0 - x1
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];
        x0.shape().broadcast(x1.shape())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = gy;
        let gx1 = neg(gy);

        let gx0 = sum_to(gx0, x0.shape());
        let gx1 = sum_to(&gx1, x1.shape());

        vec![gx0, gx1]
    }
}

impl Op for Neg {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        x.neg()
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x = x[0];
        Ok(x.shape().clone())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let gx = neg(gy);
        vec![gx]
    }
}

impl Op for Mul {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];
        x0 * x1
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];
        x0.shape().broadcast(x1.shape())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = mul(gy, x1);
        let gx1 = mul(gy, x0);

        let gx0 = sum_to(&gx0, x0.shape());
        let gx1 = sum_to(&gx1, x1.shape());

        vec![gx0, gx1]
    }
}

impl Op for Div {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];
        x0 / x1
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];
        x0.shape().broadcast(x1.shape())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = div(gy, x1);
        let gx1 = neg(&div(&mul(gy, x0), &mul(x1, x1)));

        let gx0 = sum_to(&gx0, x0.shape());
        let gx1 = sum_to(&gx1, x1.shape());

        vec![gx0, gx1]
    }
}

impl Op for Sum {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        x.sum_axis(Axis(self.axis))
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x = x[0];

        let mut d = x.shape().dim.clone();

        // we keep axis dim (i.e., keepdim=True in PyTorch)
        if d.len() > self.axis {
            d[self.axis] = 1;
            Ok(Shape::new(&d))
        } else {
            Err(ShapeError::new("cannot sum on nonexistent axis"))
        }
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x = x[0];
        let gx = broadcast_to(gy, x.shape());

        vec![gx]
    }
}

impl Op for SumTo {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        let mut y: Tensor = x.clone();
        let mut pad = vec![1, x.ndim() - self.shape.ndim()];
        pad.extend_from_slice(&self.shape.dim);

        for (d, axis) in pad.iter().rev().enumerate() {
            if d == 1 {
                y = y.sum_axis(Axis(*axis))
            }
        }
        y
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        Ok(self.shape.clone())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let gx = broadcast_to(gy, &self.shape);
        vec![gx]
    }
}

impl Op for BroadcastTo {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        // TODO: shared tensor
        x[0].clone()
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        Ok(self.shape.clone())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let gx = sum_to(gy, &self.shape);
        vec![gx]
    }
}

impl Op for MatMul {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        tensor::matmul(x0.view(), x1.view()).unwrap()
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];

        let x0_dim = &x0.shape().dim;
        let x1_dim = &x1.shape().dim;

        let x0_ndim = x0_dim.len();
        let x1_ndim = x1_dim.len();

        if x0_ndim < 2 || x1_ndim < 2 {
            return Err(ShapeError::new("invalid matrix"));
        }

        if x0_dim[x0_ndim - 1] != x1_dim[x1_ndim - 2] {
            return Err(ShapeError::new("incompatible matrix"));
        }

        let x0_bat_dim = &x0_dim[0..x0_ndim - 2];
        let x1_bat_dim = &x1_dim[0..x1_ndim - 2];

        // shape broadcast
        let mut y_dim = tensor::broadcast(x0_bat_dim, x1_bat_dim)?;

        // add matrix dim
        y_dim.push(x0_dim[x0_ndim - 2]);
        y_dim.push(x1_dim[x1_ndim - 1]);

        Ok(Shape::new(&y_dim))
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        // (*, A, B)
        let x0 = x[0];

        // (*, B, C)
        let x1 = x[1];

        // gy: (*, A, C)

        // (*, A, B) = (*, A, C) (*, C, B)
        let gx0 = matmul(gy, &transpose(x1));

        // (*, B, C) = (*, B, A) (*, A, C)
        let gx1 = matmul(&transpose(x0), gy);

        vec![gx0, gx1]
    }
}

// Swap last two components of tensor
impl Op for Transpose {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        let mut y = x.clone();

        y.swap_axes(x.ndim() - 2, x.ndim() - 1);
        y
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x = x[0];

        let mut d = x.shape().dim.clone();

        if d.len() > 1 {
            d.swap(d.len() - 1, d.len() - 2);
            Ok(Shape::new(&d))
        } else {
            Err(ShapeError::new("dim too short"))
        }
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x = x[0];
        let gx = transpose(x);
        vec![gx]
    }
}

impl Op for ReLU {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        x.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x = x[0];
        Ok(x.shape().clone())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x = x[0];
        let gx = mul(gy, &binarize(x, 0.0));
        vec![gx]
    }
}

impl Op for Binarize {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];

        x.mapv(|x| if x > self.threshold { 1.0 } else { 0.0 })
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x = x[0];
        Ok(x.shape().clone())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        unimplemented!()
    }
}

impl Op for Softmax {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];

        // for numerical stability
        let mut y: Tensor = x - *x.max().unwrap();
        y = y.mapv(|x| x.exp());
        y / y.sum_axis(Axis(self.axis))
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x = x[0];
        Ok(x.shape().clone())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x = x[0];
        let y = softmax(x, self.axis);

        let mut gx = mul(&y, gy);
        gx = sub(&gx, &mul(&y, &sum(&gx, self.axis)));

        vec![gx]
    }
}

impl Op for SoftmaxCrossEntropy {
    fn compute(&self, x: &[&Tensor]) -> Tensor {


        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

        unimplemented!()
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        unimplemented!()
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {

        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y

        unimplemented!()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

pub fn identity(x: &Var) -> Var {
    x.clone()
}

pub fn add(x0: &Var, x1: &Var) -> Var {
    op(Box::new(Add), &[x0, x1])
}

pub fn sub(x0: &Var, x1: &Var) -> Var {
    op(Box::new(Sub), &[x0, x1])
}

pub fn neg(x: &Var) -> Var {
    op(Box::new(Neg), &[x])
}

pub fn mul(x0: &Var, x1: &Var) -> Var {
    op(Box::new(Mul), &[x0, x1])
}

pub fn div(x0: &Var, x1: &Var) -> Var {
    op(Box::new(Div), &[x0, x1])
}

pub fn sum(x: &Var, axis: usize) -> Var {
    op(Box::new(Sum { axis }), &[x])
}

pub fn sum_to(x: &Var, shape: &Shape) -> Var {
    if x.shape() == shape {
        identity(x)
    } else {
        op(
            Box::new(SumTo {
                shape: shape.clone(),
            }),
            &[x],
        )
    }
}

pub fn broadcast_to(x: &Var, shape: &Shape) -> Var {
    if x.shape() == shape {
        identity(x)
    } else {
        op(
            Box::new(BroadcastTo {
                shape: shape.clone(),
            }),
            &[x],
        )
    }
}

pub fn matmul(x0: &Var, x1: &Var) -> Var {
    op(Box::new(MatMul), &[x0, x1])
}

pub fn transpose(x: &Var) -> Var {
    op(Box::new(Transpose), &[x])
}

pub fn relu(x: &Var) -> Var {
    op(Box::new(ReLU), &[x])
}

pub fn binarize(x: &Var, threshold: f32) -> Var {
    op(Box::new(Binarize { threshold }), &[x])
}

pub fn softmax(x: &Var, axis: usize) -> Var {
    op(box Softmax { axis }, &[x])
}

pub fn softmax_cross_entropy(x0: &Var, x1: &Var) -> Var {
    op(Box::new(SoftmaxCrossEntropy), &[x0, x1])
}

impl_op!(+ |a: Var, b: Var| -> op::add(&a, &b));
impl_op!(+ |a: &Var, b: Var| -> op::add(a, &b));
impl_op!(+ |a: Var, b: &Var| -> op::add(&a, b));
impl_op!(+ |a: &Var, b: &Var| -> op::add(a, b));

impl_op!(* |a: Var, b: Var| -> op::mul(&a, &b));
impl_op!(* |a: &Var, b: Var| -> op::mul(a, &b));
impl_op!(* |a: Var, b: &Var| -> op::mul(&a, b));
impl_op!(* |a: &Var, b: &Var| -> op::mul(a, b));
