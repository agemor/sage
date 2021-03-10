use crate::autodiff::ops::Operator;
use crate::autodiff::var::Var;
use crate::tensor::Tensor;

// activations
struct Relu;

struct Binarize {
    threshold: f32,
}

impl Operator<1> for Relu {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.map(|&x| if x > 0.0 { x } else { 0.0 })
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x = x[0];
        let gx = gy * binarize(x, 0.0);
        [gx]
    }
}

impl Operator<1> for Binarize {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];

        x.map(|&x| if x > self.threshold { 1.0 } else { 0.0 })
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, _x: [&Var; 1], _gy: &Var) -> [Var; 1] {
        panic!("this operation is not differentiable");
    }
}

pub fn relu(x: &Var) -> Var {
    Relu.forward([x])
}

pub fn binarize(x: &Var, threshold: f32) -> Var {
    Binarize { threshold }.forward([x])
}
