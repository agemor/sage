use crate::autodiff::{op, Op, Var};

struct Add;
struct Sub;
struct Neg;
struct Mul;
struct Div;

struct Sqrt;

impl Op for Add {
    fn compute(&self, x: &[f32]) -> f32 {
        x[0] + x[1]
    }

    fn diff(&self, _x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![gy.clone(), gy.clone()]
    }
}

impl Op for Sub {
    fn compute(&self, x: &[f32]) -> f32 {
        x[0] - x[1]
    }

    fn diff(&self, _x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![gy.clone(), -gy]
    }
}

impl Op for Neg {
    fn compute(&self, x: &[f32]) -> f32 {
        -x[0]
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![-gy]
    }
}

impl Op for Mul {
    fn compute(&self, x: &[f32]) -> f32 {
        x[0] * x[1]
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![gy * x[1], gy * x[0]]
    }
}

impl Op for Div {
    fn compute(&self, x: &[f32]) -> f32 {
        x[0] / x[1]
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![gy / x[1], -(gy * x[1]) / (x[0] * x[0])]
    }
}

impl Op for Sqrt {
    fn compute(&self, x: &[f32]) -> f32 {
        x[0].sqrt()
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![-gy / (2 * x * sqrt(x[0]))]
    }
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
