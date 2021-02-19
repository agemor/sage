use crate::autodiff::{op, Op, Var};
use crate::tensor::shape::{Dim, IntoDimension, ShapeError};
use crate::tensor::Tensor;
use std::ops;

// basic arithmetics
struct Add;
struct Sub;
struct Neg;
struct Mul;
struct Div;

// broadcasting operations
struct Sum {
    axis: usize,
}
struct SumTo {
    shape: Dim,
}
struct BroadcastTo {
    shape: Dim,
}

// matrix operations
struct MatMul;
struct MatVec;

// shaping operations
struct Transpose;
struct Reshape;
struct Select;

// activations
struct ReLU {
    inplace: bool,
}
struct Binarize {
    threshold: f32,
}

// softmax
struct Softmax {
    axis: usize,
}

// loss functions
struct SoftmaxCrossEntropy;

impl Op for Add {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];
        x0 + x1
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];

        Dim::union_k(x0.shape(), x1.shape())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = sum_to(gy, &x0.shape());
        let gx1 = sum_to(gy, &x1.shape());

        vec![gx0, gx1]
    }
}

impl Op for Sub {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];
        x0 - x1
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];
        Dim::union_k(x0.shape(), x1.shape())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = gy;
        let gx1 = neg(gy);

        let gx0 = sum_to(gx0, &x0.shape());
        let gx1 = sum_to(&gx1, &x1.shape());

        vec![gx0, gx1]
    }
}

impl Op for Neg {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        0.0 - x
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x = x[0];
        Ok(Dim::new(x.shape()))
    }

    fn backward(&self, _x: &[&Var], gy: &Var) -> Vec<Var> {
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

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];
        Dim::union_k(x0.shape(), x1.shape())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = mul(gy, x1);
        let gx1 = mul(gy, x0);

        let gx0 = sum_to(&gx0, &x0.shape());
        let gx1 = sum_to(&gx1, &x1.shape());

        vec![gx0, gx1]
    }
}

impl Op for Div {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];
        x0 / x1
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];
        Dim::union_k(x0.shape(), x1.shape())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = div(gy, x1);
        let gx1 = neg(&div(&mul(gy, x0), &mul(x1, x1)));

        let gx0 = sum_to(&gx0, &x0.shape());
        let gx1 = sum_to(&gx1, &x1.shape());

        vec![gx0, gx1]
    }
}

impl Op for Sum {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        x.sum_axis(self.axis as isize)
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x = x[0];

        let mut d = x.shape().into_dimension();

        // we keep axis dim (i.e., keepdim=True in PyTorch)
        if d.ndim() > self.axis {
            d.remove(self.axis);
            Ok(d)
        } else {
            Err(ShapeError::new("cannot sum on nonexistent axis"))
        }
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x = x[0];
        let gx = broadcast_to(gy, &x.shape());

        vec![gx]
    }
}

impl Op for SumTo {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];

        let mut pad = vec![1, x.rank() - self.shape.ndim()];
        pad.extend_from_slice(&self.shape.sizes);

        let y = pad
            .iter()
            .rev()
            .enumerate()
            .fold(x.clone(), |y, (dim_size, &axis)| {
                if dim_size == 1 {
                    y.sum_axis(axis as isize)
                } else {
                    y
                }
            });

        y
    }

    fn forward(&self, _x: &[&Var]) -> Result<Dim, ShapeError> {
        Ok(self.shape.clone())
    }

    fn backward(&self, _x: &[&Var], gy: &Var) -> Vec<Var> {
        let gx = broadcast_to(gy, &self.shape.sizes);
        vec![gx]
    }
}

impl Op for BroadcastTo {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        x.upcast(&self.shape).unwrap()
    }

    fn forward(&self, _x: &[&Var]) -> Result<Dim, ShapeError> {
        Ok(self.shape.clone())
    }

    fn backward(&self, _x: &[&Var], gy: &Var) -> Vec<Var> {
        let gx = sum_to(gy, &self.shape.sizes);
        vec![gx]
    }
}

impl Op for MatMul {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0.matmul(x1)
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];

        let x0_dim = x0.shape();
        let x1_dim = x1.shape();

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
        let mut y_dim = Dim::union_k(x0_bat_dim, x1_bat_dim)?.sizes;

        // add matrix dim
        y_dim.push(x0_dim[x0_ndim - 2]);
        y_dim.push(x1_dim[x1_ndim - 1]);

        Ok(Dim::new(&y_dim))
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

impl Op for MatVec {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0.matvec(x1)
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];

        let x0_dim = x0.shape();
        let x1_dim = x1.shape();

        let x0_ndim = x0_dim.len();
        let x1_ndim = x1_dim.len();

        if x0_ndim < 2 || x1_ndim < 1 {
            return Err(ShapeError::new("invalid matrix or vector"));
        }

        if x0_dim[x0_ndim - 1] != x1_dim[x1_ndim - 1] {
            return Err(ShapeError::new("incompatible matrix-vector"));
        }

        let x0_bat_dim = &x0_dim[0..x0_ndim - 2];
        let x1_bat_dim = &x1_dim[0..x1_ndim - 1];

        // shape broadcast
        let mut y_dim = Dim::union_k(x0_bat_dim, x1_bat_dim)?.sizes;

        // add matrix dim
        y_dim.push(x0_dim[x0_ndim - 2]);

        Ok(Dim::new(&y_dim))
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        // (*, A, B)
        let x0 = x[0];

        // (*, B)
        let x1 = x[1];

        // gy: (*, A)

        // (*, A, B) = (*, A, 1) (*, 1, B)
        let gx0 = matmul(gy, &transpose(x1));

        // (*, B) = (*, B, A) (*, A)
        let gx1 = matvec(&transpose(x0), gy);

        vec![gx0, gx1]
    }
}

// Swap last two components of tensor
impl Op for Transpose {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        x.transpose((x.rank() - 2) as isize, (x.rank() - 1) as isize)
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x = x[0];

        let mut d = x.shape().to_vec();
        let d_len = d.len();

        if d_len > 1 {
            d.swap(d_len - 1, d_len - 2);
            Ok(Dim::new(&d))
        } else {
            Err(ShapeError::new("dim too short"))
        }
    }

    fn backward(&self, x: &[&Var], _gy: &Var) -> Vec<Var> {
        let x = x[0];
        let gx = transpose(x);
        vec![gx]
    }
}

impl Op for ReLU {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];
        x.map(|&x| if x > 0.0 { x } else { 0.0 })
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x = x[0];
        Ok(x.shape().into_dimension())
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

        x.map(|&x| if x > self.threshold { 1.0 } else { 0.0 })
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x = x[0];
        Ok(x.shape().into_dimension())
    }

    fn backward(&self, _x: &[&Var], _gy: &Var) -> Vec<Var> {
        unimplemented!()
    }
}

impl Op for Softmax {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];

        //let mut reduced_shape = x.shape().to_vec();
        //reduced_shape[self.axis] = 1;

        let max = x
            .max_axis(self.axis as isize)
            .expand_dims(self.axis as isize);

        // for numerical stability
        let mut y: Tensor = x - max;
        y.mapv_inplace(|x| x.exp());

        let sum = y.sum_axis(self.axis as isize);
        y / sum
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x = x[0];
        Ok(x.shape().into_dimension())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x = x[0];
        let y = softmax(x, self.axis);

        let mut gx = mul(&y, gy);
        gx = sub(&gx, &mul(&y, &sum(&gx, self.axis)));

        vec![gx]
    }
}
// (N, C) (N, C) -> (N)
impl Op for SoftmaxCrossEntropy {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x0 = x[0];
        let t = x[1];

        let log_z = x0.log_sum_exp(1);
        let log_p = log_z * t;
        -log_p.sum_axis(1)
    }

    fn forward(&self, x: &[&Var]) -> Result<Dim, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];

        if x0.shape() != x1.shape() {
            Err(ShapeError::new("shape does not match"))
        } else {
            Ok(x0.shape().into_dimension())
        }
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x0 = x[0];
        let t = x[1];

        let sm = softmax(x0, 1);

        let gx0 = mul(&sub(&sm, t), gy);

        vec![gx0, t.clone()]
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

pub fn sum_to(x: &Var, shape: &[usize]) -> Var {
    if x.shape() == shape {
        identity(x)
    } else {
        op(
            Box::new(SumTo {
                shape: shape.into_dimension(),
            }),
            &[x],
        )
    }
}

pub fn broadcast_to(x: &Var, shape: &[usize]) -> Var {
    if x.shape() == shape {
        identity(x)
    } else {
        op(
            Box::new(BroadcastTo {
                shape: shape.into_dimension(),
            }),
            &[x],
        )
    }
}

pub fn matmul(x0: &Var, x1: &Var) -> Var {
    op(Box::new(MatMul), &[x0, x1])
}

pub fn matvec(x0: &Var, x1: &Var) -> Var {
    op(Box::new(MatVec), &[x0, x1])
}

pub fn transpose(x: &Var) -> Var {
    op(Box::new(Transpose), &[x])
}

pub fn relu(x: &Var) -> Var {
    op(Box::new(ReLU { inplace: false }), &[x])
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

impl_op!(+ |a: Var, b: Var| -> Var { add(&a, &b) });
impl_op!(+ |a: &Var, b: Var| -> Var { add(a, &b) });
impl_op!(+ |a: Var, b: &Var| -> Var { add(&a, b) });
impl_op!(+ |a: &Var, b: &Var| -> Var { add(a, b) });

impl_op!(-|a: Var, b: Var| -> Var { sub(&a, &b) });
impl_op!(-|a: &Var, b: Var| -> Var { sub(a, &b) });
impl_op!(-|a: Var, b: &Var| -> Var { sub(&a, b) });
impl_op!(-|a: &Var, b: &Var| -> Var { sub(a, b) });

impl_op!(*|a: Var, b: Var| -> Var { mul(&a, &b) });
impl_op!(*|a: &Var, b: Var| -> Var { mul(a, &b) });
impl_op!(*|a: Var, b: &Var| -> Var { mul(&a, b) });
impl_op!(*|a: &Var, b: &Var| -> Var { mul(a, b) });

impl_op!(/ |a: Var, b: Var| -> Var { div(&a, &b) });
impl_op!(/ |a: &Var, b: Var| -> Var { div(a, &b) });
impl_op!(/ |a: Var, b: &Var| -> Var { div(&a, b) });
impl_op!(/ |a: &Var, b: &Var| -> Var { div(a, b) });

impl_op!(-|a: Var| -> Var { neg(&a) });
impl_op!(-|a: &Var| -> Var { neg(a) });
