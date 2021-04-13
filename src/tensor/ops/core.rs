use crate::tensor::backend::{BinaryIndexOperation, BinaryOperation, UnaryOperation};
use crate::tensor::shape::{Shape, ShapeError, ToIndex, ToIndices, ToShape};
use crate::tensor::Tensor;
use itertools::Itertools;
use std::ops;

impl Tensor {
    pub fn binary_op(&self, other: &Tensor, op: BinaryOperation) -> Tensor {
        let union_shape = Shape::union(self.shape, other.shape).expect("cannot broadcast");

        let input1 = self.upcast(&union_shape).unwrap();
        let input2 = other.upcast(&union_shape).unwrap();
        let mut output = Tensor::uninit(union_shape, self.backend().clone());

        self.backend().binary_op(&input1, &input2, &mut output, op);

        output
    }

    pub fn unary_op(&self, op: UnaryOperation) -> Tensor {
        let mut output = Tensor::uninit(self.shape, self.backend());
        self.backend().unary_op(self, &mut output, op);
        output
    }

    pub fn reduce<Is>(&self, op: BinaryOperation, axes: Is, retain_axis: bool) -> Tensor
    where
        Is: ToIndices,
    {
        let axes = axes.to_indices(self.shape.len());

        let reduced_shape = (0..self.order())
            .filter(|a| !axes.contains(a))
            .map(|e| self.shape[e])
            .collect::<Vec<usize>>();

        let mut output = Tensor::uninit(reduced_shape, self.backend().clone());

        self.backend().reduction(self, &mut output, op, axes);

        if retain_axis {
            let shape = axes.iter().fold(self.shape, |s, a| s.replaced(a, 1));
            output.reshape(shape)
        } else {
            output
        }
    }

    pub fn reduce_index<I>(&self, op: BinaryIndexOperation, axis: I, retain_axis: bool) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.order());

        let mut output = Tensor::uninit(self.shape.removed(axis), self.backend().clone());

        self.backend().reduction_index(self, &mut output, op, axis);

        if retain_axis {
            output.expand_axis(axis)
        } else {
            output
        }
    }

    pub fn contract<Is1, Is2>(&self, other: &Tensor, axes1: Is1, axes2: Is2) -> Tensor
    where
        Is1: ToIndices,
        Is2: ToIndices,
    {
        let axes1 = axes1.to_indices(self.order());
        let axes2 = axes2.to_indices(other.order());

        let preserved_shape1 = (0..self.order())
            .filter(|a| !axes1.contains(a))
            .map(|e| self.shape[e]);

        let preserved_shape2 = (0..other.order())
            .filter(|a| !axes2.contains(a))
            .map(|e| other.shape[e]);

        let contracted_shape = preserved_shape1
            .chain(preserved_shape2)
            .collect::<Vec<usize>>();

        let mut output = Tensor::uninit(contracted_shape, self.backend().clone());

        self.backend()
            .contraction(self, other, &mut output, axes1, axes2);

        output
    }

    // * Creates new tensor
    pub fn concat<I>(inputs: &[&Tensor], axis: I) -> Tensor
    where
        I: ToIndex,
    {
        let inputs_first = inputs[0];

        let axis = axis.to_index(inputs_first.order());

        if !inputs.iter().map(|t| t.shape.removed(axis)).all_equal() {
            panic!("shape does not match");
        }

        let concat_extent = inputs.iter().map(|t| t.shape[axis]).sum();
        let concat_shape = inputs_first.shape.replaced(axis, concat_extent);

        let mut output = Tensor::uninit(concat_shape, inputs_first.backend().clone());

        inputs_first.backend().concat(inputs, &mut output, axis);

        output
    }

    // * Creates new tensor
    pub fn stack<I>(inputs: &[&Tensor], axis: I) -> Tensor
    where
        I: ToIndex,
    {
        if !inputs.iter().map(|t| t.extents()).all_equal() {
            panic!("all tensors should be in the same shape");
        }
        let inputs_first = inputs[0];

        let axis = axis.to_index(inputs_first.order() + 1);

        let stacked_shape = inputs_first.shape.inserted(axis, inputs.len());

        // convert into array views
        let expanded_inputs = inputs.iter().map(|t| t.expand_axis(axis)).collect_vec();

        Self::concat(expanded_inputs.iter().collect_vec().as_ref(), axis)
    }

    // create a new one, with default strides (= contiguous)
    pub fn recreate(&self) -> Tensor {
        let mut output = Tensor::uninit(self.shape, self.backend().clone());
        self.backend().copy(self, &mut output);
        output
    }

    pub fn squeeze_axis<I>(&self, axis: I) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.order());

        if self.shape[axis] != 1 {
            panic!("dim=1 cannot be squeezed");
        }

        let new_shape = self.shape.removed(axis);
        let mut new_strides = self.strides.clone();
        new_strides.remove(axis);

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    pub fn expand_axis<I>(&self, axis: I) -> Tensor
    where
        I: ToIndex,
    {
        // allow unexisting index
        let axis = axis.to_index(self.order() + 1);

        let new_shape = self.shape.inserted(axis, 1);
        let mut new_strides = self.strides.clone();

        if new_strides.len() == axis {
            new_strides.push(1);
        } else {
            let s = self.strides[axis];
            new_strides.insert(axis, s);
        }

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    // reshape (underlying data does not change)
    pub fn reshape<S>(&self, shape: S) -> Tensor
    where
        S: ToShape,
    {
        let new_shape = shape.to_shape(self.size());
        let new_strides = Shape::default_strides(new_shape);

        if self.is_contiguous() {
            Tensor::view(self, new_shape, &new_strides, self.offset)
        } else {
            Tensor::view(&self.recreate(), new_shape, &new_strides, 0)
        }
    }

    // swap last two dims of tensor
    pub fn transpose<I1, I2>(&self, axis1: I1, axis2: I2) -> Tensor
    where
        I1: ToIndex,
        I2: ToIndex,
    {
        let axis1 = axis1.to_index(self.order());
        let axis2 = axis2.to_index(self.order());

        if axis1 == axis2 {
            panic!("same axis");
        }

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        new_shape.swap(axis1, axis2);
        new_strides.swap(axis1, axis2);

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    pub fn permute<Is>(&self, axes: Is) -> Tensor
    where
        Is: ToIndices,
    {
        let axes = axes.to_indices(self.order());

        let mut use_counts = vec![0; self.order()];

        axes.iter().for_each(|axis| {
            use_counts[*axis] += 1;
        });

        if use_counts.iter().any(|count| *count != 1) {
            panic!("some axes are not used, or used more than once");
        }

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        for (i, axis) in axes.into_iter().enumerate() {
            new_shape[i] = self.shape[axis];
            new_strides[i] = self.strides[axis];
        }

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    pub fn upcast<S>(&self, shape: S) -> Result<Tensor, ShapeError>
    where
        S: ToShape,
    {
        let target_shape = shape.to_shape(0);

        if self.order() > target_shape.len() {
            return Err(ShapeError::new("invalid broadcast.. too small shape"));
        }

        // a = ( 10, 7)
        // 0 ... 69

        // a[3][2] = 3 * 7 + 2 * 1

        // a = (10, 1, 7)
        // 0 ... 69
        // a[3][0][2] = 3 * 7 + 0 * 1 + 2 * 1

        // a = (1, 10, 7)
        // 0 ... 69
        // a[0][3][2] = 0 * 70 + 3 * 7

        // in broadcasting we set 0 for augmented dimension.

        // create some fake strides

        // (3, 1, 5) broadcast (2, 1, 1, 9, 5)

        // padded:      (1, 1, 3, 1, 5)

        // union shape: (2, 1, 3, 9, 5)  <----- this is given as a parameter

        // stride       (0, 0, k, 0, k)
        //                        ^________ always 0 anyways...

        // pad 1 for dim

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        let pad_len = target_shape.len() - new_shape.len();

        for _ in 0..pad_len {
            new_shape.insert(0, 1);
            new_strides.insert(0, 0);
        }

        for ((a, b), s) in new_shape
            .iter_mut()
            .zip(target_shape.iter())
            .zip(new_strides.iter_mut())
        {
            if *a != *b {
                // for broadcast axes, 'mute' them by set its stride to zero
                if *a == 1 {
                    *a = *b;
                    *s = 0;
                } else {
                    return Err(ShapeError::new(
                        "invalid broadcast... target shape should be larger.",
                    ));
                }
            }
        }
        Ok(Tensor::view(self, new_shape, &new_strides, self.offset))
    }

    /// make possible a[..][3][..] kind of operations.
    pub fn index_axis<I, J>(&self, index: I, axis: J) -> Tensor
    where
        I: ToIndex,
        J: ToIndex,
    {
        let axis = axis.to_index(self.order());
        let index = index.to_index(self.shape[axis]);

        // a = (10, 10, 10)
        // a[2]
        // stride = (100, 10, 1)
        // offset = 2 * 100

        // a[2][3]
        // offset = 2 * 100 + 3 * 10

        // b = a[:][3]
        // b[7][5]
        // a offset = 3 * 10
        // b... 7 * 100 + offset + 5 * 1     ----> just remove axis

        let offset = self.strides[axis] * index;

        let mut new_shape = self.shape;
        let mut new_stride = self.strides.clone();

        new_shape.remove(axis);
        new_stride.remove(axis);

        Tensor::view(self, new_shape, &new_stride, self.offset + offset)
    }

    // single index.. unwraps first one.
    pub fn index<I>(&self, index: I) -> Tensor
    where
        I: ToIndex,
    {
        self.index_axis(index, 0)
    }

    // make possible of a[..][2..3][..] kind of operations
    pub fn slice_axis<I, J, K>(&self, start_index: I, end_index: J, axis: K) -> Tensor
    where
        I: ToIndex,
        J: ToIndex,
        K: ToIndex,
    {
        let axis = axis.to_index(self.order());

        let start_index = start_index.to_index(self.shape[axis]);
        let end_index = end_index.to_index(self.shape[axis]);

        if start_index > end_index {
            panic!("start and end index are not in the order");
        }

        // a = (10, 10, 10)
        // a[2]
        // stride = (100, 10, 1)
        // offset = 2 * 100

        // a[2][3]
        // offset = 2 * 100 + 3 * 10

        // b = a[:][3]
        // b[7][5]
        // a offset = 3 * 10
        // b... 7 * 100 + offset + 5 * 1     ----> just remove axis
        let stride = self.strides[axis];
        let offset = stride * start_index;

        let mut new_shape = self.shape;
        let mut new_stride = self.strides.clone();

        new_shape[axis] = end_index - start_index + 1;
        new_stride.remove(axis);

        Tensor::view(self, new_shape, &new_stride, self.offset + offset)
    }

    pub fn slice<I, J>(&self, start_index: I, end_index: J) -> Tensor
    where
        I: ToIndex,
        J: ToIndex,
    {
        self.slice_axis(start_index, end_index, 0)
    }
}

// math utility methods
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    a.binary_op(b, BinaryOperation::Add)
}

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    a.binary_op(b, BinaryOperation::Sub)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    a.binary_op(b, BinaryOperation::Mul)
}

pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    a.binary_op(b, BinaryOperation::Div)
}

pub fn neg(a: &Tensor) -> Tensor {
    a.unary_op(UnaryOperation::Neg)
}

macro_rules! impl_tensor_op {
    ($op:tt, $f:expr) => {
        impl_op!($op |a: Tensor, b: Tensor| -> Tensor {$f(&a, &b) });
        impl_op!($op |a: &Tensor, b: Tensor| -> Tensor {  $f(a, &b) });
        impl_op!($op |a: Tensor, b: &Tensor| -> Tensor {  $f(&a, b) });
        impl_op!($op |a: &Tensor, b: &Tensor| -> Tensor {  $f(a, b) });
        impl_op!($op |a: Tensor, b: f32| -> Tensor {  $f(&a, &Tensor::scalar(b)) });
        impl_op!($op |a: &Tensor, b: f32| -> Tensor {  $f(a, &Tensor::scalar(b))});
        impl_op!($op |a: f32, b: Tensor| -> Tensor {  $f(&Tensor::scalar(a), &b) });
        impl_op!($op |a: f32, b: &Tensor| -> Tensor {  $f(&Tensor::scalar(a), b)});
    }
}

// basic arithmetics
impl_tensor_op!(+, add);
impl_tensor_op!(-, sub);
impl_tensor_op!(*, mul);
impl_tensor_op!(/, div);

impl_op!(-|a: Tensor| -> Tensor { neg(&a) });
impl_op!(-|a: &Tensor| -> Tensor { neg(a) });

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sum() {
        let a = Tensor::from_elem([3, 10, 1], 3.0);
        assert_eq!(a.sum(), 90.0_f32);
    }

    #[test]
    fn test_sum_axis() {
        let a = Tensor::randn([3, 5, 7]);

        assert_eq!(a.sum_axis(1, false).shape(), [3, 7].to_shape(0));
        assert_eq!(a.sum_axis(-1, false), a.fold_axis(-1, 0.0, |&a, &b| a + b));
    }
}
