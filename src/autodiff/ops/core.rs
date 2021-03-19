use crate::autodiff::ops::{elemwise_comp_time, pairwise_comp_time, DebugInfo, Operator};
use crate::autodiff::var::{ToVar, Var};
use crate::profile::{torch_var, Profiler};
use crate::tensor::shape::{Shape, ToIndex, ToIndices, ToShape};
use crate::tensor::Tensor;
use std::{cmp, ops};

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

struct Permute {
    axes: Vec<usize>,
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

struct SelectMultiIndex {
    indices: Vec<usize>,
    axis: usize,
}

struct UnselectMultiIndex {
    indices: Vec<usize>,
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
pub fn benchmark_pairwise_map(x0: &Var, x1: &Var, profiler: &mut Profiler) -> usize {
    let shape = Shape::union(x0.shape(), x1.shape()).unwrap();

    let uid = format!("maps_{}", shape.to_id());

    let mut comp_time = pairwise_comp_time(1.0, x0, x1);

    if let Some(t) = profiler.comp_time(&uid) {
        comp_time = t;
    } else {
        let v1 = format!("{}1", uid);
        let v2 = format!("{}2", uid);

        profiler.add_benchmark(
            &uid,
            {
                // prep code
                format!(
                    "{}{}",
                    torch_var(&v1, x0.shape()),
                    torch_var(&v2, x1.shape())
                )
            },
            {
                // exec code
                format!("{} + {}", v1, v2)
            },
        );
    }
    comp_time
}

pub fn benchmark_elemwise_map(x: &Var, profiler: &mut Profiler) -> usize {
    let uid = format!("maps_{}", x.shape().to_id());

    let mut comp_time = elemwise_comp_time(1.0, x);

    if let Some(t) = profiler.comp_time(&uid) {
        comp_time = t;
    } else {
        let v = &uid;
        profiler.add_benchmark(
            &uid,
            {
                // prep code
                torch_var(v, x.shape())
            },
            {
                // exec code
                format!("torch.nn.functional.relu({})", v)
            },
        );
    }
    comp_time
}

impl Operator<2> for Add {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        x0 + x1
    }

    fn debug_info(&self, x: [&Var; 2], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_pairwise_map(x[0], x[1], profiler);
        DebugInfo::new("Add", y.shape().size(), comp_time)
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

    fn debug_info(&self, x: [&Var; 2], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_pairwise_map(x[0], x[1], profiler);
        DebugInfo::new("Sub", y.shape().size(), comp_time)
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

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_elemwise_map(x[0], profiler);

        DebugInfo::new("Neg", y.shape().size(), comp_time)
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

    fn debug_info(&self, x: [&Var; 2], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_pairwise_map(x[0], x[1], profiler);
        DebugInfo::new("Mul", y.shape().size(), comp_time)
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0: &Var = x[0];
        let x1 = x[1];

        if let Err(_) = Shape::union(x0.shape(), x1.shape()) {
            //println!("{}, {}", x0.debug_info().unwrap(), x1.debug_info().unwrap());
            println!("{}, {}", x0.shape(), x1.shape());
        }

        let shape = Shape::union(x0.shape(), x1.shape()).unwrap();
        Var::from_binary_op(shape, self, [x0, x1])
    }

    fn is_fdb(&self) -> bool {
        true
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

    fn debug_info(&self, x: [&Var; 2], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_pairwise_map(x[0], x[1], profiler);
        DebugInfo::new("Div", y.shape().size(), comp_time)
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        let shape = Shape::union(x0.shape(), x1.shape()).unwrap();
        Var::from_binary_op(shape, self, [x0, x1])
    }

    fn is_fdb(&self) -> bool {
        true
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

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_elemwise_map(x[0], profiler);
        DebugInfo::new("ScalarAdd", y.shape().size(), comp_time)
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

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_elemwise_map(x[0], profiler);
        DebugInfo::new("ScalarSub", y.shape().size(), comp_time)
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

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_elemwise_map(x[0], profiler);
        DebugInfo::new("ScalarMul", y.shape().size(), comp_time)
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

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let comp_time = benchmark_elemwise_map(x[0], profiler);
        DebugInfo::new("ScalarDiv", y.shape().size(), comp_time)
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

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let uid = format!("sum_{}", x[0].shape().to_id());

        let mut comp_time = elemwise_comp_time(x[0].shape()[self.axis] as f32, x[0]);

        if let Some(t) = profiler.comp_time(&uid) {
            comp_time = t;
        } else {
            let v1 = &uid;

            profiler.add_benchmark(
                &uid,
                {
                    // prep code
                    torch_var(v1, x[0].shape())
                },
                {
                    // exec code
                    format!("torch.sum({}, dim={})", v1, self.axis)
                },
            );
        }

        DebugInfo::new("Sum", y.shape().size(), comp_time)
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

        let gx = if self.retain_axis {
            gy.broadcast_to(x.shape())
        } else {
            gy.unsqueeze(self.axis).broadcast_to(x.shape())
        };

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

    fn debug_info(&self, x: [&Var; 1], y: &Var, profiler: &mut Profiler) -> DebugInfo {
        let uid = format!("sumto_{}", x[0].shape().to_id());

        let mut comp_time = elemwise_comp_time(1.0, x[0]);

        if let Some(t) = profiler.comp_time(&uid) {
            comp_time = t;
        } else {
            let v1 = &uid;

            profiler.add_benchmark(
                &uid,
                {
                    // prep code
                    torch_var(v1, x[0].shape())
                },
                {
                    // exec code
                    format!("torch.sum({}, dim=({}))", v1, self.shape.to_string2())
                },
            );
        }

        DebugInfo::new("SumTo", y.shape().size(), comp_time)
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

    fn debug_info(&self, x: [&Var; 1], _y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("BroadcastTo", x[0].shape().size(), 1)
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

        x.reshape(self.to).unwrap()
    }

    fn debug_info(&self, x: [&Var; 1], _y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("Reshape", x[0].shape().size(), 1)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];

        // check shape compatibility
        if x.shape().size() != self.from.size() || self.from.size() != self.to.size() {
            panic!("incompatible size");
        }

        Var::from_unary_op(self.to, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.reshape(self.from);
        [gx]
    }
}

// Reshape variable
impl Operator<1> for Permute {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];

        x.permute(self.axes.as_slice())
    }

    fn debug_info(&self, x: [&Var; 1], _y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("Permute", x[0].shape().size(), 1)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x = x[0];
        let mut shape = x.shape();
        shape.permute(self.axes.as_slice());

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        // simple argsort
        let reverse = (0..self.axes.len())
            .into_iter()
            .map(|i| {
                // must unwrap
                self.axes.iter().position(|&axis| axis == i).unwrap()
            })
            .collect::<Vec<usize>>();

        let gx = gy.permute(reverse.as_slice());
        [gx]
    }
}

// select index from variable
impl Operator<1> for SelectIndex {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];

        x.index_axis(self.index, self.axis)
    }

    fn debug_info(&self, _x: [&Var; 1], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("SelectIndex", y.shape().size(), 1)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x: &Var = x[0];

        let mut shape = x.shape();
        shape.replace(self.axis, 1);

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x = x[0];
        let orig_size = x.shape()[self.axis];
        let gx = gy.unselect_index(self.index, orig_size, self.axis);
        [gx]
    }
}

impl Operator<1> for UnselectIndex {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        unimplemented!();
    }

    fn debug_info(&self, _x: [&Var; 1], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("UnselectIndex", y.shape().size(), 1)
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

    fn debug_info(&self, _x: [&Var; 1], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("SelectSlice", y.shape().size(), 1)
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

    fn debug_info(&self, _x: [&Var; 1], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("UnselectSlice", y.shape().size(), 1)
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

impl Operator<1> for SelectMultiIndex {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];

        unimplemented!();
    }

    fn debug_info(&self, _x: [&Var; 1], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("SelectMultiIndex", y.shape().size(), 1)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x: &Var = x[0];

        let mut shape = x.shape();
        shape.replace(self.axis, self.indices.len());

        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let x = x[0];
        let orig_size = x.shape()[self.axis];
        let gx = gy.unselect_multi_index(self.indices.as_slice(), orig_size, self.axis);
        [gx]
    }
}

impl Operator<1> for UnselectMultiIndex {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        unimplemented!();
    }

    fn debug_info(&self, _x: [&Var; 1], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("UnselectMultiIndex", y.shape().size(), 1)
    }

    fn forward(self, x: [&Var; 1]) -> Var {
        let x: &Var = x[0];

        let mut shape = x.shape();
        if shape[self.axis] != self.indices.len() {
            panic!("invalid target axis size");
        }
        shape.replace(self.axis, self.size);
        Var::from_unary_op(shape, self, x)
    }

    fn backward(&self, _x: [&Var; 1], gy: &Var) -> [Var; 1] {
        let gx = gy.multi_index(self.indices.as_slice(), self.axis);
        [gx]
    }
}

impl Operator<2> for Concat {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        Tensor::cat(&[x0, x1], self.axis).unwrap()
    }

    fn debug_info(&self, x: [&Var; 2], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("Concat", y.shape().size(), 1)
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

impl Operator<2> for Stack {
    fn compute(&self, x: [&Tensor; 2]) -> Tensor {
        let x0 = x[0];
        let x1 = x[1];

        Tensor::stack(&[x0, x1], self.axis).unwrap()
    }

    fn debug_info(&self, x: [&Var; 2], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("Stack", y.shape().size(), 1)
    }

    fn forward(self, x: [&Var; 2]) -> Var {
        let x0 = x[0];
        let x1 = x[1];

        // shapes of two variable must be identical
        if x0.shape() == x1.shape() {
            panic!("invalid stack shape");
        }

        let mut shape = x0.shape();
        shape.insert(self.axis, 2);

        Var::from_binary_op(shape, self, [x0, x1])
    }

    fn backward(&self, x: [&Var; 2], gy: &Var) -> [Var; 2] {
        let x0: &Var = x[0];
        let x1: &Var = x[1];

        let gx0 = gy.index(0, self.axis);
        let gx1 = gy.index(1, self.axis);

        [gx0, gx1]
    }
}

impl Operator<1> for Expand {
    fn compute(&self, x: [&Tensor; 1]) -> Tensor {
        let x = x[0];
        x.expand_dims(self.axis)
    }

    fn debug_info(&self, _x: [&Var; 1], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("Expand", y.shape().size(), 1)
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

    fn debug_info(&self, _x: [&Var; 1], y: &Var, _profiler: &mut Profiler) -> DebugInfo {
        DebugInfo::new("Squeeze", y.shape().size(), 1)
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
    pub fn scalar_add(&self, v: f32) -> Var {
        scalar_add(self, v)
    }

    pub fn scalar_sub(&self, v: f32) -> Var {
        scalar_sub(self, v)
    }

    pub fn scalar_mul(&self, v: f32) -> Var {
        scalar_mul(self, v)
    }

    pub fn scalar_div(&self, v: f32) -> Var {
        scalar_div(self, v)
    }

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
        let shape = shape.to_shape(0);
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
        let shape = shape.to_shape(0);
        if self.shape() == shape {
            self.clone()
        } else {
            BroadcastTo { shape }.forward([self])
        }
    }

    pub fn reshape<S>(&self, shape: S) -> Var
    where
        S: ToShape,
    {
        Reshape {
            from: self.shape(),
            to: shape.to_shape(self.shape().size()),
        }
        .forward([self])
    }

    pub fn permute<Is>(&self, axes: Is) -> Var
    where
        Is: ToIndices,
    {
        Permute {
            axes: axes.to_indices(self.rank()),
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
        .forward([self, &v])
    }

    pub fn stack<V, I>(&self, other: V, axis: I) -> Var
    where
        V: ToVar,
        I: ToIndex,
    {
        Stack {
            axis: axis.to_index(self.rank() + 1),
        }
        .forward([self, &other.to_var()])
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
            index: index.to_index(size),
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
            index: index.to_index(size),
            slice_size,
            size,
            axis,
        }
        .forward([self])
    }

    pub fn multi_index<I, J>(&self, indices: I, axis: J) -> Var
    where
        I: ToIndices,
        J: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        SelectMultiIndex {
            indices: indices.to_indices(self.shape()[axis]),
            axis,
        }
        .forward([self])
    }

    fn unselect_multi_index<I, J>(&self, indices: I, size: usize, axis: J) -> Var
    where
        I: ToIndices,
        J: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        UnselectMultiIndex {
            indices: indices.to_indices(size),
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

        let v = var.squeeze(2).unsqueeze(2);
        assert!(v.data().equals(&var.data(), 0.001));
    }
}
