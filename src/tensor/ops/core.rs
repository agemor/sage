use crate::tensor::shape::{Shape, ShapeError, ToIndex, ToIndices, ToShape};
use crate::tensor::Tensor;
use itertools::Itertools;
use num_traits::FromPrimitive;

impl Tensor {
    // * Creates new tensor
    pub fn cat<I>(tensors: &[&Tensor], axis: I) -> Result<Tensor, ShapeError>
    where
        I: ToIndex,
    {
        let axis = axis.to_index(tensors[0].rank());

        // TODO: spit out some errors when tensors in different shape, except in the cat axis.

        // convert into array views
        let t_data = tensors
            .iter()
            .map(|t| t.to_ndarray())
            .collect::<Vec<ndarray::ArrayViewD<f32>>>();

        if let Ok(arr) = ndarray::concatenate(ndarray::Axis(axis), &t_data) {
            Ok(Tensor::from_ndarray(arr))
        } else {
            Err(ShapeError::new("cannot concat!"))
        }
    }

    // * Creates new tensor
    pub fn stack<I>(tensors: &[&Tensor], axis: I) -> Result<Tensor, ShapeError>
    where
        I: ToIndex,
    {
        if !tensors.iter().map(|e| e.shape).all_equal() {
            panic!("all tensors should be in the same shape");
        }

        let axis = axis.to_index(tensors[0].rank() + 1);

        // convert into array views
        let t_data = tensors
            .iter()
            .map(|t| t.to_ndarray())
            .collect::<Vec<ndarray::ArrayViewD<f32>>>();
        if let Ok(arr) = ndarray::stack(ndarray::Axis(axis), &t_data) {
            Ok(Tensor::from_ndarray(arr))
        } else {
            Err(ShapeError::new("cannot stack!"))
        }
    }

    pub fn squeeze<I>(&self, axis: I) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        if new_shape[axis] == 1 {
            new_shape.remove(axis);
            new_strides.remove(axis);
        } else {
            panic!("dim=1 cannot be squeezed");
        }
        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    pub fn expand_dims<I>(&self, axis: I) -> Tensor
    where
        I: ToIndex,
    {
        // allow unexisting index
        let axis = axis.to_index(self.rank() + 1);

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        new_shape.insert(axis, 1);

        if new_strides.len() == axis {
            new_strides.push(1);
        } else {
            let s = self.strides[axis];
            new_strides.insert(axis, s);
        }

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    // reshape (underlying data does not change)
    pub fn reshape<S>(&self, shape: S) -> Result<Tensor, ShapeError>
    where
        S: ToShape,
    {
        let new_shape = shape.to_shape(self.size());
        let new_strides = Shape::default_strides(new_shape);

        if self.is_contiguous() {
            Ok(Tensor::view(self, new_shape, &new_strides, self.offset))
        } else {
            Err(ShapeError::new("tensor not contiguous"))
        }
    }

    // swap last two dims of tensor
    pub fn transpose<I, J>(&self, axis_a: I, axis_b: J) -> Tensor
    where
        I: ToIndex,
        J: ToIndex,
    {
        let axis_a = axis_a.to_index(self.rank());
        let axis_b = axis_b.to_index(self.rank());

        if axis_a == axis_b {
            panic!("same axis");
        }

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        new_shape.swap(axis_a, axis_b);
        new_strides.swap(axis_a, axis_b);

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    pub fn permute<Is>(&self, axes: Is) -> Tensor
    where
        Is: ToIndices,
    {
        let axes = axes.to_indices(self.rank());

        let mut use_counts = vec![0; self.rank()];

        axes.iter().for_each(|axis| {
            use_counts[*axis] += 1;
        });

        if use_counts.iter().any(|count| *count != 1) {
            panic!("some axes are not used, or used more than once");
        }

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        for axis in axes {
            new_shape[axis] = self.shape[axis];
            new_strides[axis] = self.strides[axis];
        }

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    pub fn upcast<S>(&self, shape: S) -> Result<Tensor, ShapeError>
    where
        S: ToShape,
    {
        let target_shape = shape.to_shape(0);

        if self.rank() > target_shape.len() {
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
        let axis = axis.to_index(self.rank());
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
        let axis = axis.to_index(self.rank());

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

    pub fn sum(&self) -> f32 {
        self.logical_iter().sum()
    }

    pub fn sum_axis<I>(&self, axis: I, retain_axis: bool) -> Tensor
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());

        let mut new_shape = self.shape;
        new_shape.remove(axis);

        let mut summed = Tensor::zeros(new_shape);

        for t in self.along_axis(axis) {
            summed = summed + t;
        }

        if retain_axis {
            summed.expand_dims(axis)
        } else {
            summed
        }
    }
}

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
