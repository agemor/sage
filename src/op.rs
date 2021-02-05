use crate::autodiff::{op, Op, Tensor, Var};
use std::ops;
use std::ops::Neg as Neg2;

struct Add;
struct Sub;
struct Neg;
struct Mul;
struct Div;

struct Sqrt;

struct Sum;
struct SumTo {
    shape: Vec<usize>,
}
struct BroadcastTo {
    shape: Vec<usize>,
}

impl Op for Add {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        x[0] + x[1]
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![gy.clone(), gy.clone()]

        if x[0].shape

    }
}

impl Op for Sub {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        x[0] - x[1]
    }

    fn diff(&self, _x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![gy.clone(), -gy]
    }
}

impl Op for Neg {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        x[0].neg()
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![-gy]
    }
}

impl Op for Mul {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        x[0] * x[1]
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![gy * x[1], gy * x[0]]
    }
}

impl Op for Div {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        x[0] / x[1]
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![gy / x[1], -(gy * x[1]) / (x[0] * x[0])]
    }
}

impl Op for Sqrt {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        x[0].mapv(f32::sqrt)
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![-gy / (2 * x * sqrt(x[0]))]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

impl Op for Sum {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        unimplemented!()
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        unimplemented!()
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

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![broadcast_to(gy, &self.shape)]
    }
}

impl Op for BroadcastTo {
    fn compute(&self, x: &[&Tensor]) -> Tensor {
        // This behavior is automatic
        x[0].clone()
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![sum_to(gy, &self.shape)]
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

pub fn sqrt(x: &Var) -> Var {
    op(Box::new(Sqrt), &[x])
}

pub fn sum_to(x: &Var, shape: &[usize]) -> Var {
    //TODO: pass identity if shape is same
    op(
        Box::new(SumTo {
            shape: shape.to_vec(),
        }),
        &[x],
    )
}

pub fn broadcast_to(x: &Var, shape: &[usize]) -> Var {
    //TODO: pass identity if shape is same
    op(
        Box::new(BroadcastTo {
            shape: shape.to_vec(),
        }),
        &[x],
    )
}

impl_op!(+ |a: Var, b: Var| -> op::add(&a, &b));
impl_op!(+ |a: &Var, b: Var| -> op::add(a, &b));
impl_op!(+ |a: Var, b: &Var| -> op::add(&a, b));
impl_op!(+ |a: &Var, b: &Var| -> op::add(a, b));

impl_op!(* |a: Var, b: Var| -> op::mul(&a, &b));
impl_op!(* |a: &Var, b: Var| -> op::mul(a, &b));
impl_op!(* |a: Var, b: &Var| -> op::mul(&a, b));
impl_op!(* |a: &Var, b: &Var| -> op::mul(a, b));
