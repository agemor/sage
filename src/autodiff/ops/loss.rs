use crate::autodiff::ops::Operator;
use crate::autodiff::var::{ToVar, Var};
use crate::tensor::Tensor;

// loss functions
struct SoftmaxCrossEntropy;

// (N, C) (N, C) -> (N)
impl Operator<2> for SoftmaxCrossEntropy {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let t = x[1];

        let log_z = x0 - x0.log_sum_exp(1, true); // ln10
        let log_p = log_z * t;
        -log_p.sum_axis(1, false)
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        if x0.shape() != x1.shape() {
            panic!("shape does not match");
        }

        let mut shape = x0.shape();
        shape.remove(-1);

        Var::from_binary_op(shape, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        let x0 = x[0];
        let t = x[1];

        let sm = x0.softmax(1);

        let n = x0.shape()[0] as f32;
        // y: [N]
        // (N, k) - (N, k) * (N, 1)

        let gx0 = (sm - t) * gy.unsqueeze(1) / n;
        [gx0, t.clone()]
    }
}

pub fn softmax_cross_entropy<V, W>(x: V, t: W) -> Var
where
    V: ToVar,
    W: ToVar,
{
    SoftmaxCrossEntropy.forward([&x.to_var(), &t.to_var()])
}
