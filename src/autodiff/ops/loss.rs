use crate::autodiff::ops::{elemwise_comp_time, pairwise_comp_time, DebugInfo, Operator};
use crate::autodiff::var::{ToVar, Var};
use crate::profile::Profiler;
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

    fn debug_info(&self, x: [&Var; 2], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("SoftmaxCrossEntropy", y.shape().size(), 1)
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

    fn is_fdb(&self) -> bool {
        true
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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::autodiff::diff;

    #[test]
    fn test_softmax_cross_entropy() {
        let input_data = Tensor::from_slice(
            [5, 10],
            &[
                0.0681, -0.4750, -0.1068, 0.2453, -0.5245, 0.1971, 0.0826, -0.4771, 0.7162,
                -1.5326, -2.1222, 2.6529, 0.1163, 2.4620, -0.3893, -0.7439, -0.1908, -0.2767,
                1.4722, 0.2627, 0.7419, 0.3707, 0.0854, 0.3992, -2.4740, -0.9155, -0.7988, 0.1836,
                -0.3489, 0.1029, -0.4769, 0.6530, 0.8418, 0.6481, 0.1508, 0.9778, 2.2582, 0.8823,
                -0.2821, 1.3810, -0.4457, 2.3899, 0.3116, 1.1650, 0.4207, 1.6690, -1.9891, -0.2580,
                0.6080, -1.3612,
            ],
        );

        let input_grad_data = Tensor::from_slice(
            [5, 10],
            &[
                -0.1778, 0.0129, 0.0186, 0.0265, 0.0123, 0.0252, 0.0225, 0.0129, 0.0424, 0.0045,
                0.0007, -0.1202, 0.0063, 0.0660, 0.0038, 0.0027, 0.0046, 0.0043, 0.0245, 0.0073,
                0.0417, 0.0287, -0.1784, 0.0296, 0.0017, 0.0079, 0.0089, 0.0238, 0.0140, 0.0220,
                0.0045, 0.0141, 0.0170, -0.1860, 0.0085, 0.0195, 0.0701, 0.0177, 0.0055, 0.0291,
                0.0049, 0.0841, 0.0105, 0.0247, -0.1883, 0.0409, 0.0011, 0.0060, 0.0142, 0.0020,
            ],
        );

        let label_data = Tensor::from_slice(
            [5, 10],
            &[
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 1., 0., 0., 0., 0., 0.,
            ],
        );
        let loss_data = Tensor::from_slice([5], &[2.1987, 0.9184, 2.2250, 2.6592, 2.8357]);

        let input = Var::with_data(input_data);
        let label = Var::with_data(label_data);

        let loss = softmax_cross_entropy(&input, &label);

        // forward check
        assert!(loss.data().equals(&loss_data, 0.001));

        let grads = diff(&loss, &[&input]);

        let input_grad = grads.get(&input).unwrap();

        // backward check
        assert!(input_grad.data().equals(&input_grad_data, 0.001));
    }
}
