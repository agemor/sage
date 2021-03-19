use crate::autodiff::ops::core::benchmark_elemwise_map;
use crate::autodiff::ops::{elemwise_comp_time, DebugInfo, Operator};
use crate::autodiff::var::Var;
use crate::profile::Profiler;
use crate::tensor::shape::ToIndex;
use crate::tensor::Tensor;

// softmax
struct Softmax {
    axis: usize,
}

impl Operator<1> for Softmax {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];

        x.softmax(self.axis)
    }

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_elemwise_map(x[0], profiler);
        DebugInfo::new("Softmax", y.shape().size(), comp_time)
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
        let y = x.softmax(self.axis);

        let mut gx = &y * gy;
        gx = &gx - y * gx.sum(self.axis, true);

        [gx]
    }
}

impl Var {
    pub fn mean<I>(&self, axis: I, retain_axis: bool) -> Var
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());
        self.sum(axis, retain_axis) / self.shape()[axis] as f32
    }

    pub fn var<I>(&self, axis: I, retain_axis: bool) -> Var
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());
        (self - self.mean(axis, true))
            .pow(2.0)
            .sum(axis, retain_axis)
            / (self.shape()[axis] - 1) as f32
    }
    pub fn softmax<I>(&self, axis: I) -> Var
    where
        I: ToIndex,
    {
        Softmax {
            axis: axis.to_index(self.rank()),
        }
        .forward([self])
    }
}
