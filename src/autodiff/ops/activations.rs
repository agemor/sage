use crate::autodiff::ops::core::{add_bench_elemwise_map, comp_cost_elemwise};
use crate::autodiff::ops::{elemwise_comp_time, DebugInfo, Operator};
use crate::autodiff::var::{ToVar, Var};
use crate::profile::{torch_var, Profiler};
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

// activations
struct Relu;

struct LeakyRelu {
    alpha: f32,
}

struct Sigmoid;

struct Tanh;

struct Binarize {
    threshold: f32,
}

// trait TorchProfile {
//
//     fn uid(&self) -> String;
//
//     fn prep(&self, vars:Vec<Shape>) -> String ;
//
//     fn exec_code(&self, vars:Vec<Shape>) -> String;
// }

impl Operator<1> for Sigmoid {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.map(|&x| if x > 0.0 { x } else { 0.0 })
    }

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &Profiler) -> DebugInfo {
        let comp_time = comp_cost_elemwise(x[0], profiler);
        DebugInfo::new("Sigmoid", y.shape().size(), comp_time)
    }

    fn add_bench(&self, x: [&Var; 1], profiler: &mut Profiler) {
        add_bench_elemwise_map(x[0], profiler);
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

impl Operator<1> for Tanh {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.map(|&x| if x > 0.0 { x } else { 0.0 })
    }
    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &Profiler) -> DebugInfo {
        let comp_time = comp_cost_elemwise(x[0], profiler);
        DebugInfo::new("Tanh", y.shape().size(), comp_time)
    }

    fn add_bench(&self, x: [&Var; 1], profiler: &mut Profiler) {
        add_bench_elemwise_map(x[0], profiler);
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

impl Operator<1> for Relu {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.map(|&x| if x > 0.0 { x } else { 0.0 })
    }

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &Profiler) -> DebugInfo {
        let comp_time = comp_cost_elemwise(x[0], profiler);
        DebugInfo::new("Relu", y.shape().size(), comp_time)
    }

    fn add_bench(&self, x: [&Var; 1], profiler: &mut Profiler) {
        add_bench_elemwise_map(x[0], profiler);
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

impl Operator<1> for LeakyRelu {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.map(|&x| if x > 0.0 { x } else { 0.0 })
    }

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &Profiler) -> DebugInfo {
        let comp_time = comp_cost_elemwise(x[0], profiler);
        DebugInfo::new("LeakyRelu", y.shape().size(), comp_time)
    }

    fn add_bench(&self, x: [&Var; 1], profiler: &mut Profiler) {
        add_bench_elemwise_map(x[0], profiler);
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

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &Profiler) -> DebugInfo {
        let comp_time = comp_cost_elemwise(x[0], profiler);
        DebugInfo::new("Binarize", y.shape().size(), comp_time)
    }

    fn add_bench(&self, x: [&Var; 1], profiler: &mut Profiler) {
        add_bench_elemwise_map(x[0], profiler);
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, _x: [&Var; 1], _gy: &Var) -> [Var; 1] {
        panic!("this operation is not differentiable");
    }
}

pub fn relu<V>(x: V) -> Var
where
    V: ToVar,
{
    Relu.forward([&x.to_var()])
}

pub fn leaky_relu<V>(x: V, alpha: f32) -> Var
where
    V: ToVar,
{
    LeakyRelu { alpha }.forward([&x.to_var()])
}

pub fn sigmoid<V>(x: V) -> Var
where
    V: ToVar,
{
    Sigmoid.forward([&x.to_var()])
}

pub fn tanh<V>(x: V) -> Var
where
    V: ToVar,
{
    Tanh.forward([&x.to_var()])
}

pub fn binarize(x: &Var, threshold: f32) -> Var {
    Binarize { threshold }.forward([x])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::diff;

    #[test]
    fn test_relu() {
        let input_data = Tensor::from_slice(
            [3, 7],
            &[
                -0.61592, -0.28000, -0.67419, 0.13923, -0.43073, 0.81796, -0.27639, -1.21364,
                -0.21150, 0.35873, -0.43101, 1.25069, -2.20712, 0.58242, -0.28829, -1.22030,
                0.32220, 1.27633, -1.18069, -0.25927, -1.47341,
            ],
        );
        let output_data = Tensor::from_slice(
            [3, 7],
            &[
                0.00000, 0.00000, 0.00000, 0.13923, 0.00000, 0.81796, 0.00000, 0.00000, 0.00000,
                0.35873, 0.00000, 1.25069, 0.00000, 0.58242, 0.00000, 0.00000, 0.32220, 1.27633,
                0.00000, 0.00000, 0.00000,
            ],
        );

        let input_grad_data = Tensor::from_slice(
            [3, 7],
            &[
                0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
            ],
        );

        let input = Var::with_data(input_data);
        let output = relu(&input);

        // forward check
        assert!(output.data().equals(&output_data, 0.001));

        let grads = diff(&output, &[&input]);

        let input_grad = grads.get(&input).unwrap();

        // backward check
        assert!(input_grad.data().equals(&input_grad_data, 0.0001));
    }
}
