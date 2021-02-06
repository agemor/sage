use crate::autodiff::{op, Op, Var};
use crate::tensor::{Shape, ShapeError, Tensor};
use std::alloc::Global;
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
    threshold: usize,
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
        -x
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
        x.sum(self.axis)
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x = x[0];

        let mut d = x.shape().dim.clone();

        // we keep axis dim (i.e., keepdim=True in PyTorch)
        if d.len() > self.axis {
            d[self.axis] = 1;
            Ok(Shape::new(&d))
        } else {
            Err(ShapeError)
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
        let mut pad = vec![1, x.shape().len() - self.shape.len()];
        pad.extend_from_slice(&self.shape);

        for (d, axis) in pad.iter().rev().enumerate() {
            if d == 1 {
                y = y.sum_axis(ndarray::Axis(*axis))
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
        // This behavior is automatic
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

        x0.dot(x1)
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x0 = x[0];
        let x1 = x[1];

        let d0 = &x0.shape().dim;
        let d1 = &x1.shape().dim;

        // (j, 1, n, m) * (k, m, p) = (j, k, n, p)
        if d0.len() > 1 && d1.len() > 1 {
            let (ld0, rd0) = d0.split_at(d0.len() - 1);
            let (ld1, rd1) = d1.split_at(d1.len() - 2);

            // m == m
            if rd0[0] == rd1[0] {
                let mut sld0 = Shape::new(ld0);
                let mut sld1 = Shape::new(ld1);

                // (k) -> (k, 1)
                sld1.dim.push(1);

                // (j, 1, n) and (k, 1) -> (j, k, n)
                sld0.broadcast(&sld1);

                // (j, k, n) -> (j, k, n, p)
                sld0.dim.push(rd1[1]);

                Ok(sld0)
            } else {
                Err(ShapeError)
            }
        } else {
            Err(ShapeError)
        }
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        // (m)
        let x0 = x[0];

        // (m, k)
        let x1 = x[1];

        // (m, k) * (k, m)
        let gx0 = matmul(gy, &transpose(x1));
        let gx1 = matmul(&transpose(x0), gy);

        vec![gx0, gx1]
    }
}

impl Op for Transpose {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];

        x.transpose()
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x = x[0];

        let mut d = x.shape().dim.clone();

        if d.len() > 1 {
            d.swap(d.len() - 1, d.len() - 2);
            Ok(Shape::new(&d))
        } else {
            Err(ShapeError)
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

        x.mapv(max)
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        let x = x[0];
        Ok(x.shape().clone())
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        let x = x[0];

        let gx = mul(gy, &binarize(x, 0));

        vec![gx]
    }
}

impl Op for Binarize {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        let x = x[0];

        x.mapv(binarize, self.threshold)
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

        let y = x - x.max(self.axis);
        let y = xp.exp(y);
        y / y.sum(self.axis)
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
        unimplemented!()
    }

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        unimplemented!()
    }

    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
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

pub fn binarize(x: &Var, threshold: usize) -> Var {
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
