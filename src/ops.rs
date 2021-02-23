use crate::autodiff::{Operator, Var};
use crate::tensor::Tensor;
use std::ops;
use crate::shape::{Shape, ToIndex, ToShape};

// basic arithmetics
struct Add;

struct Sub;

struct Neg;

struct Mul;

struct Div;

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

// matrix operations
struct Matmul;

struct Matvec;

// shaping operations
struct Transpose {
    axis_a: usize,
    axis_b: usize,
}

struct Reshape;

struct Select;

struct Expand { axis: usize }

struct Squeeze { axis: usize }

// activations
struct Relu;

struct Binarize {
    threshold: f32,
}

// softmax
struct Softmax {
    axis: usize,
}

// loss functions
struct SoftmaxCrossEntropy;

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

        let gx0 = sum_to(gy, x0.shape());
        let gx1 = sum_to(gy, x1.shape());

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

        let gx0 = sum_to(gy, &x0.shape());
        let gx1 = sum_to(&-gy, &x1.shape());

        [gx0, gx1]
    }
}

impl Operator<1> for Neg {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        -x
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        Var::from_unary_op(x[0].shape(), self, x[0])
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

        let gx0 = sum_to(&(gy * x1), x0.shape());
        let gx1 = sum_to(&(gy * x0), x1.shape());

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

        let gx0 = sum_to(&(gy / x1), &x0.shape());
        let gx1 = sum_to(&(-(gy * x0) / (x1 * x1)), &x1.shape());

        [gx0, gx1]
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
        let gx = broadcast_to(gy, x.shape());

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
        let gx = broadcast_to(gy, self.shape);
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
        let gx = sum_to(gy, self.shape);
        [gx]
    }
}

impl Operator<2> for Matmul {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0.matmul(x1)
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        if x0.rank() < 2 || x1.rank() < 2 {
            panic!("should provide a matrix");
        }

        if x0.shape()[x0.rank() - 1] != x1.shape()[x1.rank() - 2] {
            panic!("matrix not compatible");
        }

        let (x0_batch, _) = x0.shape().split(x0.rank() - 2);
        let (x1_batch, _) = x1.shape().split(x1.rank() - 2);

        // shape broadcast
        let mut batch = Shape::union(x0_batch, x1_batch).unwrap();

        // add matrix dim
        batch.insert(-1, x0.shape()[x0.rank() - 2]);
        batch.insert(-1, x1.shape()[x1.rank() - 1]);

        Var::from_binary_op(batch, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        // (*, A, B)
        let x0 = x[0];

        // (*, B, C)
        let x1 = x[1];

        // gy: (*, A, C)

        // (*, A, B) = (*, A, C) (*, C, B)
        let gx0 = sum_to(&matmul(gy, &transpose(x1, -1, -2)), x0.shape());

        // (*, B, C) = (*, B, A) (*, A, C)
        let gx1 = sum_to(&matmul(&transpose(x0, -1, -2), gy), x1.shape());

        [gx0, gx1]
    }
}

impl Operator<2> for Matvec {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0.matvec(x1)
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        if x0.rank() < 2 || x1.rank() < 1 {
            panic!("invalid matrix or vector");
        }

        if x0.shape()[x0.rank() - 1] != x1.shape()[x1.rank() - 1] {
            panic!("incompatible matrix-vector");
        }

        let (x0_batch, _) = x0.shape().split(x0.rank() - 2);
        let (x1_batch, _) = x1.shape().split(x1.rank() - 1);

        // shape broadcast
        let mut batch = Shape::union(x0_batch, x1_batch).unwrap();

        // add matrix dim
        batch.insert(-1, x0.shape()[x0.rank() - 2]);

        Var::from_binary_op(batch, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        // (*, A, B)
        let x0 = x[0];

        // (*, B)
        let x1 = x[1];

        // gy: (*, A)

        // (*, A, B) = (*, A, 1) (*, 1, B)
        let gx0 = sum_to(&matmul(&expand(gy, gy.rank()), &expand(x1, x0.rank() - 1)), x0.shape());

        // (*, B) = (*, B, A) (*, A)
        let gx1 = sum_to(&matvec(&transpose(x0, -1, -2), gy), x1.shape());

        [gx0, gx1]
    }
}

// Swap last two components of tensor
impl Operator<1> for Transpose {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.transpose(self.axis_a, self.axis_b)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];

        if x.rank() < 2 {
            panic!("cannot transpose on a vector");
        }

        let mut shape = x.shape();
        shape.swap(self.axis_a, self.axis_b);

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = transpose(gy, self.axis_b, self.axis_a);
        [gx]
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
        let gx = squeeze(gy, self.axis);
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
        let gx = expand(gy, self.axis);
        [gx]
    }
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
        let gx = mul(gy, &binarize(x, 0.0));
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
        let y = softmax(x, self.axis);

        let mut gx = &y * gy;
        gx = &gx - &y * &sum(&gx, self.axis, true);

        [gx]
    }
}

// (N, C) (N, C) -> (N)
impl Operator<2> for SoftmaxCrossEntropy {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let t = x[1];

        let log_z = x0.log_sum_exp(1);

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

        let sm = softmax(x0, -1);

        let gx0 = (sm - t) * expand(gy, 1);
        // does not calculate adjoint for t. btw, who need them?
        [gx0, t.clone()]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

pub fn identity(x: &Var) -> Var {
    x.clone()
}

pub fn add(a: &Var, b: &Var) -> Var {
    Add.forward([a, b])
}

pub fn sub(a: &Var, b: &Var) -> Var {
    Sub.forward([a, b])
}

pub fn neg(x: &Var) -> Var {
    Neg.forward([x])
}

pub fn mul(a: &Var, b: &Var) -> Var {
    Mul.forward([a, b])
}

pub fn div(a: &Var, b: &Var) -> Var {
    Div.forward([a, b])
}

pub fn sum<I>(x: &Var, axis: I, retain_axis: bool) -> Var
    where I: ToIndex
{
    Sum { axis: axis.to_index(x.rank()), retain_axis }.forward([x])
}

pub fn sum_to<S>(x: &Var, shape: S) -> Var
    where S: ToShape
{
    let shape = shape.to_shape();
    if x.shape() == shape {
        identity(x)
    } else {
        SumTo { shape }.forward([x])
    }
}

pub fn broadcast_to<S>(x: &Var, shape: S) -> Var
    where S: ToShape
{
    let shape = shape.to_shape();
    if x.shape() == shape {
        identity(x)
    } else {
        BroadcastTo { shape }.forward([x])
    }
}

pub fn matmul(x0: &Var, x1: &Var) -> Var {
    Matmul.forward([x0, x1])
}

pub fn matvec(x0: &Var, x1: &Var) -> Var {
    Matvec.forward([x0, x1])
}

pub fn transpose<I, J>(x: &Var, axis_a: I, axis_b: J) -> Var
    where I: ToIndex, J: ToIndex
{
    Transpose {
        axis_a: axis_a.to_index(x.rank()),
        axis_b: axis_b.to_index(x.rank()),
    }.forward([x])
}

pub fn expand<I>(x: &Var, axis: I) -> Var
    where I: ToIndex
{
    Expand { axis: axis.to_index(x.rank() + 1) }.forward([x])
}

pub fn squeeze<I>(x: &Var, axis: I) -> Var
    where I: ToIndex
{
    Squeeze { axis: axis.to_index(x.rank()) }.forward([x])
}

pub fn relu(x: &Var) -> Var {
    Relu.forward([x])
}

pub fn binarize(x: &Var, threshold: f32) -> Var {
    Binarize { threshold }.forward([x])
}

pub fn softmax<I>(x: &Var, axis: I) -> Var
    where I: ToIndex
{
    Softmax { axis: axis.to_index(x.rank()) }.forward([x])
}

pub fn softmax_cross_entropy(x: &Var, t: &Var) -> Var {
    SoftmaxCrossEntropy.forward([x, t])
}

impl_op!(+ |a: Var, b: Var| -> Var { add(&a, &b) });
impl_op!(+ |a: &Var, b: Var| -> Var { add(a, &b) });
impl_op!(+ |a: Var, b: &Var| -> Var { add(&a, b) });
impl_op!(+ |a: &Var, b: &Var| -> Var { add(a, b) });

impl_op!(-|a: Var, b: Var| -> Var { sub(&a, &b) });
impl_op!(-|a: &Var, b: Var| -> Var { sub(a, &b) });
impl_op!(-|a: Var, b: &Var| -> Var { sub(&a, b) });
impl_op!(-|a: &Var, b: &Var| -> Var { sub(a, b) });

impl_op!(*|a: Var, b: Var| -> Var { mul(&a, &b) });
impl_op!(*|a: &Var, b: Var| -> Var { mul(a, &b) });
impl_op!(*|a: Var, b: &Var| -> Var { mul(&a, b) });
impl_op!(*|a: &Var, b: &Var| -> Var { mul(a, b) });

impl_op!(/ |a: Var, b: Var| -> Var { div(&a, &b) });
impl_op!(/ |a: &Var, b: Var| -> Var { div(a, &b) });
impl_op!(/ |a: Var, b: &Var| -> Var { div(&a, b) });
impl_op!(/ |a: &Var, b: &Var| -> Var { div(a, b) });

impl_op!(-|a: Var| -> Var { neg(&a) });
impl_op!(-|a: &Var| -> Var { neg(a) });


////////////// unit tests //////////////



