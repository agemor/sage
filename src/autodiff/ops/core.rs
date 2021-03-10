use crate::autodiff::ops::Operator;
use crate::autodiff::var::{ToVar, Var};
use crate::tensor::shape::{Shape, ToIndex, ToShape};
use crate::tensor::Tensor;
use std::cmp;

// basic arithmetics
struct Add;

struct Sub;

struct Neg;

struct Mul;

struct Div;

struct ScalarAdd {
    scalar: f32,
}

struct ScalarSub {
    scalar: f32,
}

struct ScalarMul {
    scalar: f32,
}

struct ScalarDiv {
    scalar: f32,
}

// broadcasting operations
struct Sum {
    axis: usize,
    retain_axis: bool,
}

struct SumTo {
    shape: Shape,
}

struct BroadcastTo {
    shape: Shape,
}

struct Reshape {
    from: Shape,
    to: Shape,
}

struct SelectIndex {
    index: usize,
    axis: usize,
}

struct UnselectIndex {
    index: usize,
    size: usize,
    axis: usize,
}

struct SelectSlice {
    index: usize,
    slice_size: usize,
    axis: usize,
}

struct UnselectSlice {
    index: usize,
    slice_size: usize,
    size: usize,
    axis: usize,
}

struct Concat {
    axis: usize,
}

struct Stack {
    axis: usize,
}

struct Expand {
    axis: usize,
}

struct Squeeze {
    axis: usize,
}

// math operations

impl Operator<2> for Add {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0 + x1
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        let shape = Shape::union(x0.shape(), x1.shape()).unwrap();
        Var::from_binary_op(shape, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = gy.sum_to(x0.shape());
        let gx1 = gy.sum_to(x1.shape());

        [gx0, gx1]
    }
}

impl Operator<2> for Sub {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0 - x1
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        let shape = Shape::union(x0.shape(), x1.shape()).unwrap();
        Var::from_binary_op(shape, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = gy.sum_to(x0.shape());
        let gx1 = -gy.sum_to(x1.shape());

        [gx0, gx1]
    }
}

impl Operator<1> for Neg {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        -x
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = neg(gy);
        [gx]
    }
}

impl Operator<2> for Mul {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0 * x1
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        let shape = Shape::union(x0.shape(), x1.shape()).unwrap();
        Var::from_binary_op(shape, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = (gy * x1).sum_to(x0.shape());
        let gx1 = (gy * x0).sum_to(x1.shape());

        [gx0, gx1]
    }
}

impl Operator<2> for Div {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];
        x0 / x1
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        let shape = Shape::union(x0.shape(), x1.shape()).unwrap();
        Var::from_binary_op(shape, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        let x0 = x[0];
        let x1 = x[1];

        let gx0 = (gy / x1).sum_to(x0.shape());
        let gx1 = (-(gy * x0) / (x1 * x1)).sum_to(x1.shape());

        [gx0, gx1]
    }
}

impl Operator<1> for ScalarAdd {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x + self.scalar
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.clone();
        [gx]
    }
}

impl Operator<1> for ScalarSub {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x - self.scalar
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.clone();
        [gx]
    }
}

impl Operator<1> for ScalarMul {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x * self.scalar
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = scalar_mul(gy, self.scalar);
        [gx]
    }
}

impl Operator<1> for ScalarDiv {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x / self.scalar
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        Var::from_unary_op(x.shape(), self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = scalar_div(gy, self.scalar);
        [gx]
    }
}

impl Operator<1> for Sum {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.sum_axis(self.axis, self.retain_axis)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];

        let mut shape = x.shape();

        if self.retain_axis {
            shape.replace(self.axis, 1);
        } else {
            shape.remove(self.axis);
        }

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x = x[0];
        let gx = gy.broadcast_to(x.shape());

        [gx]
    }
}

impl Operator<1> for SumTo {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];

        let mut shape = self.shape;
        let d = x.rank() - self.shape.len();

        for _ in 0..d {
            shape.insert(0, 1);
        }

        let mut y = shape
            .iter()
            .enumerate()
            .fold(x.clone(), |y, (axis, &dim_size)| {
                if dim_size == 1 {
                    y.sum_axis(axis, true)
                } else {
                    y
                }
            });

        for _ in 0..d {
            y = y.squeeze(0);
        }
        y
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];

        assert!(x.shape().len() >= self.shape.len());
        assert!(x.shape().size() >= self.shape.size());

        // assert compatibility
        Shape::union(x.shape(), self.shape).unwrap();

        Var::from_unary_op(self.shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.broadcast_to(self.shape);
        [gx]
    }
}

impl Operator<1> for BroadcastTo {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];

        x.upcast(self.shape).unwrap()
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];

        assert!(x.shape().len() <= self.shape.len());
        assert!(x.shape().size() <= self.shape.size());

        // assert compatibility
        Shape::union(x.shape(), self.shape).unwrap();

        Var::from_unary_op(self.shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.sum_to(self.shape);
        [gx]
    }
}

// Reshape variable
impl Operator<1> for Reshape {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];

        x.reshape(self.to)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];

        // check shape compatibility
        if x.shape().size() != self.from.size() || self.from.size() != self.to.size() {
            panic!("incompatible size");
        }

        Var::from_unary_op(self.to.size(), self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.reshape(self.from);
        [gx]
    }
}

// select index from variable
impl Operator<1> for SelectIndex {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];

        x.index_axis(self.index, self.axis)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x: &Var = x[0];

        let mut shape = x.shape();
        shape.replace(self.axis, 1);

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x: &Var = x[0];
        let orig_size = x.shape()[self.axis];
        let gx = gy.unselect_index(self.index, orig_size, self.axis);
        [gx]
    }
}

impl Operator<1> for UnselectIndex {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        unimplemented!();
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x: &Var = x[0];

        let mut shape = x.shape();
        if shape[self.axis] != 1 {
            panic!("invalid target axis size");
        }
        shape.replace(self.axis, self.size);
        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.index(self.index, self.axis);
        [gx]
    }
}

// select index from variable
impl Operator<1> for SelectSlice {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x: &Tensor = x[0];

        x.slice_axis(self.index, self.index + self.slice_size, self.axis)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x: &Var = x[0];

        let mut shape = x.shape();
        shape.replace(self.axis, self.slice_size);

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x: &Var = x[0];
        let orig_size = x.shape()[self.axis];
        let gx = gy.unselect_slice(self.index, self.slice_size, orig_size, self.axis);
        [gx]
    }
}

// select index from variable
impl Operator<1> for UnselectSlice {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x: &Tensor = x[0];
        unimplemented!();
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x: &Var = x[0];

        let mut shape = x.shape();
        shape.replace(self.axis, self.size);

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.slice(self.index, self.slice_size, self.axis);
        [gx]
    }
}

impl Operator<2> for Concat {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        Tensor::cat(&[x0, x1], self.axis).unwrap()
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        // shapes of two variable must be identical
        let mut shape0 = x0.shape();
        let mut shape1 = x1.shape();

        if x0.shape().remove(self.axis) != shape1.remove(self.axis) {
            panic!("invalid concat shape");
        }

        let mut shape = x0.shape();
        shape.replace(self.axis, x0.shape()[self.axis] + x1.shape()[self.axis]);

        Var::from_binary_op(shape, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        let x0: &Var = x[0];
        let x1: &Var = x[1];

        let x0_slice_size = x0.shape()[self.axis];
        let x1_slice_size = x1.shape()[self.axis];

        let gx0 = gy.slice(0, x0_slice_size, self.axis);
        let gx1 = gy.slice(x0_slice_size, x1_slice_size, self.axis);

        [gx0, gx1]
    }
}

impl Operator<1> for Expand {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.expand_dims(self.axis)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        let mut shape = x.shape();
        shape.insert(self.axis, 1);

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.squeeze(self.axis);
        [gx]
    }
}

impl Operator<1> for Squeeze {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.squeeze(self.axis)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        let mut shape = x.shape();
        shape.remove(self.axis);

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.unsqueeze(self.axis);
        [gx]
    }
}

pub fn add<V: ToVar, W: ToVar>(a: V, b: W) -> Var {
    Add.forward([&a.to_var(), &b.to_var()])
}

pub fn sub<V: ToVar, W: ToVar>(a: V, b: W) -> Var {
    Sub.forward([&a.to_var(), &b.to_var()])
}

pub fn neg<V: ToVar>(x: V) -> Var {
    Neg.forward([&x.to_var()])
}

pub fn mul<V: ToVar, W: ToVar>(a: V, b: W) -> Var {
    Mul.forward([&a.to_var(), &b.to_var()])
}

pub fn div<V: ToVar, W: ToVar>(a: V, b: W) -> Var {
    Div.forward([&a.to_var(), &b.to_var()])
}

pub fn scalar_add<V: ToVar>(x: V, scalar: f32) -> Var {
    ScalarAdd { scalar }.forward([&x.to_var()])
}

pub fn scalar_sub<V: ToVar>(x: V, scalar: f32) -> Var {
    ScalarSub { scalar }.forward([&x.to_var()])
}

pub fn scalar_mul<V: ToVar>(x: V, scalar: f32) -> Var {
    ScalarMul { scalar }.forward([&x.to_var()])
}

pub fn scalar_div<V: ToVar>(x: V, scalar: f32) -> Var {
    ScalarDiv { scalar }.forward([&x.to_var()])
}

impl Var {
    pub fn sum<I>(&self, axis: I, retain_axis: bool) -> Var
    where
        I: ToIndex,
    {
        Sum {
            axis: axis.to_index(self.rank()),
            retain_axis,
        }
        .forward([self])
    }

    pub fn sum_to<S>(&self, shape: S) -> Var
    where
        S: ToShape,
    {
        let shape = shape.to_shape();
        if self.shape() == shape {
            self.clone()
        } else {
            SumTo { shape }.forward([self])
        }
    }

    pub fn broadcast_to<S>(&self, shape: S) -> Var
    where
        S: ToShape,
    {
        let shape = shape.to_shape();
        if self.shape() == shape {
            self.clone()
        } else {
            BroadcastTo { shape }.forward([self])
        }
    }

    pub fn transpose<I, J>(&self, axis_a: I, axis_b: J) -> Var
    where
        I: ToIndex,
        J: ToIndex,
    {
        Transpose {
            axis_a: axis_a.to_index(self.rank()),
            axis_b: axis_b.to_index(self.rank()),
        }
        .forward([self])
    }

    pub fn reshape<S>(&self, shape: S) -> Var
    where
        S: ToShape,
    {
        Reshape {
            from: self.shape(),
            to: shape.to_shape(),
        }
        .forward([self])
    }

    pub fn concat<V, I>(&self, other: V, axis: I) -> Var
    where
        V: ToVar,
        I: ToIndex,
    {
        let v = other.to_var();
        Concat {
            axis: axis.to_index(cmp::min(self.rank(), v.rank())),
        }
        .forward([self, &v1])
    }

    pub fn index<I, J>(&self, index: I, axis: J) -> Var
    where
        I: ToIndex,
        J: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        SelectIndex {
            index: index.to_index(self.shape()[axis]),
            axis,
        }
        .forward([self])
    }

    fn unselect_index<I, J>(&self, index: I, size: usize, axis: J) -> Var
    where
        I: ToIndex,
        J: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        UnselectIndex {
            index: index.to_index(self.shape()[axis]),
            size,
            axis,
        }
        .forward([self])
    }

    pub fn slice<I, J>(&self, index: I, slice_size: usize, axis: J) -> Var
    where
        I: ToIndex,
        J: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        SelectSlice {
            index: index.to_index(self.shape()[axis]),
            slice_size,
            axis,
        }
        .forward([self])
    }

    fn unselect_slice<I, J>(&self, index: I, slice_size: usize, size: usize, axis: J) -> Var
    where
        I: ToIndex,
        J: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        UnselectSlice {
            index: index.to_index(self.shape()[axis]),
            slice_size,
            size,
            axis,
        }
        .forward([self])
    }

    pub fn squeeze<I>(&self, axis: I) -> Var
    where
        I: ToIndex,
    {
        Squeeze {
            axis: axis.to_index(self.rank()),
        }
        .forward([self])
    }

    pub fn unsqueeze<I>(&self, axis: I) -> Var
    where
        I: ToIndex,
    {
        Expand {
            axis: axis.to_index(self.rank() + 1),
        }
        .forward([self])
    }
}

impl_op!(+ |a: Var, b: Var| -> Var { add(a, b) });
impl_op!(+ |a: &Var, b: Var| -> Var { add(a, b) });
impl_op!(+ |a: Var, b: &Var| -> Var { add(a, b) });
impl_op!(+ |a: &Var, b: &Var| -> Var { add(a, b) });

impl_op!(+|a: Var, b: f32| -> Var { scalar_add(a, b) });
impl_op!(+|a: &Var, b: f32| -> Var { scalar_add(a, b) });
impl_op!(+|a: f32, b: Var| -> Var { scalar_add(b, a) });
impl_op!(+|a: f32, b: &Var| -> Var { scalar_add(b, a) });

impl_op!(-|a: Var, b: Var| -> Var { sub(a, b) });
impl_op!(-|a: &Var, b: Var| -> Var { sub(a, b) });
impl_op!(-|a: Var, b: &Var| -> Var { sub(a, b) });
impl_op!(-|a: &Var, b: &Var| -> Var { sub(a, b) });

impl_op!(-|a: Var, b: f32| -> Var { scalar_sub(a, b) });
impl_op!(-|a: &Var, b: f32| -> Var { scalar_sub(a, b) });
impl_op!(-|a: f32, b: Var| -> Var { scalar_sub(b, a) });
impl_op!(-|a: f32, b: &Var| -> Var { scalar_sub(b, a) });

impl_op!(*|a: Var, b: Var| -> Var { mul(a, b) });
impl_op!(*|a: &Var, b: Var| -> Var { mul(a, b) });
impl_op!(*|a: Var, b: &Var| -> Var { mul(a, b) });
impl_op!(*|a: &Var, b: &Var| -> Var { mul(a, b) });

impl_op!(*|a: Var, b: f32| -> Var { scalar_mul(a, b) });
impl_op!(*|a: &Var, b: f32| -> Var { scalar_mul(a, b) });
impl_op!(*|a: f32, b: Var| -> Var { scalar_mul(b, a) });
impl_op!(*|a: f32, b: &Var| -> Var { scalar_mul(b, a) });

impl_op!(/ |a: Var, b: Var| -> Var { div(a, b) });
impl_op!(/ |a: &Var, b: Var| -> Var { div(a, b) });
impl_op!(/ |a: Var, b: &Var| -> Var { div(a, b) });
impl_op!(/ |a: &Var, b: &Var| -> Var { div(a, b) });

impl_op!(/|a: Var, b: f32| -> Var { scalar_div(a, b) });
impl_op!(/|a: &Var, b: f32| -> Var { scalar_div(a, b) });
impl_op!(/|a: f32, b: Var| -> Var { scalar_div(b, a) });
impl_op!(/|a: f32, b: &Var| -> Var { scalar_div(b, a) });

impl_op!(-|a: Var| -> Var { neg(a) });
impl_op!(-|a: &Var| -> Var { neg(a) });

////////////// unit tests //////////////

#[cfg(test)]
mod tests {

    use super::*;
    use crate::autodiff::diff;

    #[test]
    fn test_squeeze_and_expand() {
        let data = Tensor::randn([3, 5, 1, 4]);
        let var = Var::with_data(data);

        let v = expand(&squeeze(&var, 2), 2);
        assert!(v.data().equals(&var.data(), 0.001));
    }

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

    #[test]
    fn test_matmul() {
        let a_data = Tensor::from_slice(
            [2, 3, 5],
            &[
                0.59291, -0.93341, -0.18223, 0.59965, 0.00716, -0.46003, -0.42066, -0.64802,
                0.55233, 1.73416, -0.22048, 0.19131, -1.55627, 0.71593, 0.46732, -1.04307,
                -0.35595, 1.09157, 1.20627, 1.08326, 0.99519, -1.14578, 0.43299, -1.93212,
                -0.79032, 0.45944, -0.53933, -0.36932, 0.57457, -0.82304,
            ],
        );

        let a_grad_data = Tensor::from_slice(
            [2, 3, 5],
            &[
                -0.07216, -2.58671, 2.36674, 0.32271, 1.93365, -0.07216, -2.58671, 2.36674,
                0.32271, 1.93365, -0.07216, -2.58671, 2.36674, 0.32271, 1.93365, -0.07216,
                -2.58671, 2.36674, 0.32271, 1.93365, -0.07216, -2.58671, 2.36674, 0.32271, 1.93365,
                -0.07216, -2.58671, 2.36674, 0.32271, 1.93365,
            ],
        );

        let b_data = Tensor::from_slice(
            [5, 4],
            &[
                -0.07156, 0.44339, -0.14176, -0.30223, -0.12564, -1.72730, -0.69673, -0.03704,
                1.63514, -0.71166, 0.25041, 1.19285, 0.26871, -0.66543, -0.07882, 0.79825, 0.03763,
                0.64892, 0.37141, 0.87569,
            ],
        );

        let b_grad_data = Tensor::from_slice(
            [5, 4],
            &[
                0.32396, 0.32396, 0.32396, 0.32396, -3.20382, -3.20382, -3.20382, -3.20382,
                -1.23128, -1.23128, -1.23128, -1.23128, 1.71663, 1.71663, 1.71663, 1.71663,
                1.67854, 1.67854, 1.67854, 1.67854,
            ],
        );

        let c_data = Tensor::from_slice(
            [2, 3, 4],
            &[
                -0.06173, 1.61048, 0.47605, 0.12295, -0.76016, 1.74160, 0.79658, 1.34111, -2.34301,
                0.50618, -0.37460, -0.81613, 2.26913, -0.72422, 0.97646, 3.54202, 0.23182, 2.88506,
                0.62440, -1.97623, -0.44558, 0.48170, -0.13282, -0.82150,
            ],
        );

        let a = Var::with_data(a_data);
        let b = Var::with_data(b_data);

        let c = matmul(&a, &b);

        // forward check

        assert!(c.data().equals(&c_data, 0.001));

        let grads = diff(&c, &[&a, &b]);

        let a_grad = grads.get(&a).unwrap();
        let b_grad = grads.get(&b).unwrap();

        // backward check
        assert!(a_grad.data().equals(&a_grad_data, 0.001));
        assert!(b_grad.data().equals(&b_grad_data, 0.001));
    }

    #[test]
    fn test_matvec() {
        let a_data = Tensor::from_slice(
            [2, 3, 5],
            &[
                -3.07866, -0.79024, -0.94439, 1.67945, 1.27508, -0.09492, 1.62357, 2.64593,
                1.74636, 0.82562, 1.41111, 0.05571, -0.48702, -0.46361, 0.35668, -1.28441,
                -0.91176, -1.09115, -0.19053, 1.63678, 2.29281, 1.03232, 1.15872, 1.36020,
                -0.72647, 1.21023, 1.16816, -1.05799, 0.13244, -0.97486,
            ],
        );

        let a_grad_data = Tensor::from_slice(
            [2, 3, 5],
            &[
                2.17630, 0.64114, 0.94362, 0.66252, -0.90203, 2.17630, 0.64114, 0.94362, 0.66252,
                -0.90203, 2.17630, 0.64114, 0.94362, 0.66252, -0.90203, 2.17630, 0.64114, 0.94362,
                0.66252, -0.90203, 2.17630, 0.64114, 0.94362, 0.66252, -0.90203, 2.17630, 0.64114,
                0.94362, 0.66252, -0.90203,
            ],
        );

        let b_data = Tensor::from_slice([5], &[2.17630, 0.64114, 0.94362, 0.66252, -0.90203]);

        let b_grad_data = Tensor::from_slice([5], &[0.45616, 2.17776, 0.22410, 4.26431, 2.39283]);

        let c_data = Tensor::from_slice(
            [2, 3],
            &[-8.13538, 3.74338, 2.01827, -6.01211, 8.30155, 3.35153],
        );

        let a = Var::with_data(a_data);
        let b = Var::with_data(b_data);

        let c = matvec(&a, &b);

        // forward check
        assert!(c.data().equals(&c_data, 0.001));

        let grads = diff(&c, &[&a, &b]);

        let a_grad = grads.get(&a).unwrap();
        let b_grad = grads.get(&b).unwrap();

        // backward check
        assert!(a_grad.data().equals(&a_grad_data, 0.001));
        assert!(b_grad.data().equals(&b_grad_data, 0.001));
    }

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
