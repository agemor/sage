use crate::autodiff::{op, Op, Var};
use std::alloc::Global;

struct Add;
struct Sub;
struct Neg;
struct Div;
struct Mul;

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
        unimplemented!()
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var, Global> {
        unimplemented!()
    }
}

impl Op for Div {
    fn compute(&self, x: &[f32]) -> f32 {
        unimplemented!()
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var, Global> {
        unimplemented!()
    }
}

impl Op for Mul {
    fn compute(&self, x: &[f32]) -> f32 {
        unimplemented!()
    }

    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var, Global> {
        unimplemented!()
    }
}

impl Op for Sqrt {
    // All functions should be

    fn compute(&self, x: &[f32]) -> f32 {
        // Yields real value
        x[0].sqrt()
    }
    // Optimization profiles

    // Outputs variable.. for higher order derivatives
    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var> {
        vec![Var::new(1.0)]
    }
}

fn sqrt(x: &Var) -> Var {
    op(Box::new(Sqrt), &[x])
}
