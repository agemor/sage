use crate::autodiff::ops::Operator;
use crate::autodiff::var::Var;
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

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x = x[0];
        let y = x.softmax(self.axis);

        let mut gx = y * gy;
        gx = gx - y * gx.sum(self.axis, true);

        [gx]
    }
}

impl Var {
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
