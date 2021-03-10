pub mod activations;
mod conv;
pub mod core;
pub mod linalg;
pub mod loss;
pub mod math;
pub mod stats;

use crate::autodiff::Var;
use crate::tensor::shape::{Shape, ToIndex, ToShape};
use crate::tensor::Tensor;
use std::ops::Deref;
use std::{cmp, ops};

pub trait Operator<const N: usize> {
    fn compute(&self, x: [&Tensor; N]) -> Tensor;

    fn forward(self, x: [&Var; N]) -> Var;

    // f(x) = u
    // f'(x) -> ∂u/∂x
    // f'(x) * gy = ∂u/∂x * ∂y/∂u = ∂y/∂x
    fn backward(&self, x: [&Var; N], gy: &Var) -> [Var; N];
}

pub struct Operation<const N: usize> {
    operator: Box<dyn Operator<{ N }>>,
    order: usize,

    input: [Var; N],
    output: WeakVar, // to prevent cyclic references
}

impl<const N: usize> Operation<{ N }> {
    pub fn new<O>(operator: O, input: [Var; N], output: Var) -> Self
    where
        O: Operator<{ N }> + 'static,
    {
        let order = input
            .iter()
            .map(|a| a.node().order()) // each rank
            .max() // get max rank in parent gen
            .unwrap()
            + 1; // increase + 1

        Operation {
            operator: Box::new(operator),
            order,
            input,
            output: output.to_weak(),
        }
    }

    pub fn compute(&self) -> Tensor {
        let data = self.input.each_ref().map(|v| v.data());
        let data_borrowed = data.each_ref().map(|t| t.deref());
        self.operator.compute(data_borrowed)
    }

    pub fn mem_req(&self) -> usize {
        self.operator.mem_req()
    }

    pub fn input(&self) -> [Var; N] {
        self.input.clone()
    }

    pub fn output(&self) -> Var {
        self.output.to_var().unwrap()
    }

    pub fn input_adjoint(&self, output_adjoint: &Var) -> [Var; N] {
        let input = self.input.each_ref();
        self.operator.backward(input, output_adjoint)
    }
}

pub enum OperationEnum {
    Unary(Operation<1>),
    Binary(Operation<2>),
}

// TODO: write some macros
impl OperationEnum {
    pub fn arity(&self) -> usize {
        match self {
            Self::Unary(_) => 1,
            Self::Binary(_) => 2,
        }
    }

    pub fn order(&self) -> usize {
        match self {
            Self::Unary(o) => o.order,
            Self::Binary(o) => o.order,
        }
    }

    pub fn compute(&self) -> Tensor {
        match self {
            Self::Unary(o) => o.compute(),
            Self::Binary(o) => o.compute(),
        }
    }

    // returns time and memory consumption
    pub fn complexity(&self) -> Complexity {
        match self {
            Self::Unary(o) => o.complexity(),
            Self::Binary(o) => o.complexity(),
        }
    }

    pub fn mem_req(&self) -> usize {
        match self {
            Self::Unary(o) => o.mem_req(),
            Self::Binary(o) => o.mem_req(),
        }
    }

    pub fn input(&self) -> Vec<Var> {
        // TODO: change it to smallvec for better performance
        match self {
            Self::Unary(o) => o.input.to_vec(),
            Self::Binary(o) => o.input.to_vec(),
        }
    }

    pub fn output(&self) -> Option<Var> {
        match self {
            Self::Unary(o) => o.output.to_var(),
            Self::Binary(o) => o.output.to_var(),
        }
    }

    pub fn input_adjoint(&self, output_adjoint: &Var) -> Vec<Var> {
        match self {
            Self::Unary(o) => o.input_adjoint(output_adjoint).to_vec(),
            Self::Binary(o) => o.input_adjoint(output_adjoint).to_vec(),
        }
    }
}
