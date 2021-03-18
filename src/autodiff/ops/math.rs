use crate::autodiff::ops::core::scalar_mul;
use crate::autodiff::ops::{elemwise_comp_time, DebugInfo, Operator};
use crate::autodiff::var::Var;
use crate::tensor::Tensor;

struct Recip;

struct Sqrt;

struct Pow {
    n: f32,
}

impl Operator<1> for Recip {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.recip()
    }

    fn debug_info(&self, x: [&Var; 1], y: &Var) -> DebugInfo {
        DebugInfo::new("Recip", y.shape().size(), elemwise_comp_time(1.0, x[0]))
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn is_fdb(&self) -> bool {
        true
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x = x[0];
        let gx = gy * x.pow(-2.0).scalar_mul(-2.0);
        [gx]
    }
}

impl Operator<1> for Sqrt {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.sqrt()
    }

    fn debug_info(&self, x: [&Var; 1], y: &Var) -> DebugInfo {
        DebugInfo::new("Sqrt", y.shape().size(), elemwise_comp_time(1.0, x[0]))
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn is_fdb(&self) -> bool {
        true
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x = x[0];
        let gx = gy / x.sqrt().scalar_mul(2.0);
        [gx]
    }
}

impl Operator<1> for Pow {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.pow(self.n)
    }

    fn debug_info(&self, x: [&Var; 1], y: &Var) -> DebugInfo {
        DebugInfo::new("Pow", y.shape().size(), elemwise_comp_time(1.0, x[0]))
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn is_fdb(&self) -> bool {
        true
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x = x[0];
        let gx = gy * x.pow(self.n - 1.0).scalar_mul(self.n - 1.0);
        [gx]
    }
}

impl Var {
    pub fn recip(&self) -> Var {
        Recip.forward([self])
    }

    pub fn sqrt(&self) -> Var {
        Sqrt.forward([self])
    }

    pub fn pow(&self, n: f32) -> Var {
        Pow { n }.forward([self])
    }
}
