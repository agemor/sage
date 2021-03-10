use crate::autodiff::ops::core::scalar_mul;
use crate::autodiff::ops::Operator;
use crate::autodiff::var::Var;
use crate::tensor::Tensor;

struct Reciprocal;

struct Sqrt;

struct Pow;

impl Operator<1> for Reciprocal {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        unimplemented!()
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        unimplemented!()
    }
}

impl Operator<1> for Sqrt {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        unimplemented!()
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x = x[0];
        let gx = gy / scalar_mul(x.sqrt(), 2.0);
        [gx]
    }
}

impl Operator<1> for Pow {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        unimplemented!()
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        unimplemented!()
    }
}

impl Var {
    pub fn reciprocal(&self) -> Var {
        Reciprocal.forward([self])
    }

    pub fn sqrt(&self) -> Var {
        Sqrt.forward([self])
    }

    pub fn pow(&self) -> Var {
        Pow.forward([self])
    }
}
