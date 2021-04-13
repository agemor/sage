use crate::tensor::backend::{BinaryIndexOperation, BinaryOperation};
use crate::tensor::shape::{ToIndex, ToIndices};
use crate::tensor::Tensor;

impl Tensor {
    pub fn sum<Is>(&self, axes: Is, retain_axis: bool) -> Tensor
    where
        Is: ToIndices,
    {
        self.reduce(BinaryOperation::Add, axes, retain_axis)
    }

    pub fn max<Is>(&self, axes: Is, retain_axis: bool) -> Tensor
    where
        Is: ToIndices,
    {
        self.reduce(BinaryOperation::Max, axes, retain_axis)
    }

    pub fn min<Is>(&self, axes: Is, retain_axis: bool) -> Tensor
    where
        Is: ToIndices,
    {
        self.reduce(BinaryOperation::Min, axes, retain_axis)
    }

    pub fn mean<Is>(&self, axes: Is, retain_axis: bool) -> Tensor
    where
        Is: ToIndices,
    {
        let axes = axes.to_indices(self.order());
        let size = axes.iter().map(|a| self.shape[a]).product::<usize>();
        self.sum(axes, retain_axis) / size as f32
    }

    pub fn variance<Is>(&self, axes: Is, retain_axis: bool) -> Tensor
    where
        Is: ToIndices,
    {
        let axes = axes.to_indices(self.order());
        let size = axes.iter().map(|a| self.shape[a]).product::<usize>();

        (self - self.mean(axes.as_ref(), true))
            .pow(2.0)
            .sum(axes.as_ref(), retain_axis)
            / (size - 1) as f32
    }

    pub fn argmax<I>(&self, axis: I, retain_axis: bool) -> Tensor
    where
        I: ToIndex,
    {
        self.reduce_index(BinaryIndexOperation::Max, axis, retain_axis)
    }

    pub fn argmin<I>(&self, axis: I, retain_axis: bool) -> Tensor
    where
        I: ToIndex,
    {
        self.reduce_index(BinaryIndexOperation::Min, axis, retain_axis)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_max_axis() {
        let a = Tensor::from_elem([3, 2, 5], 10.0);
        let b = Tensor::from_elem([3, 2, 5], 3.0);

        // (3, 4, 5)
        let c = Tensor::concat(&[&a, &b], 1).unwrap();

        assert_eq!(c.max_axis(1, true), Tensor::from_elem([3, 1, 5], 10.0));
    }

    #[test]
    fn test_argmax() {
        let a = Tensor::from_slice(
            [3, 5],
            &[
                0.37894, -1.43962, -0.03472, 1.50011, 1.10574, 1.20776, -0.74392, -0.10786,
                0.48039, -0.82024, -0.62761, -0.94768, 0.75950, 1.23026, 1.93393,
            ],
        );

        let b = Tensor::from_slice([5], &[1., 1., 2., 0., 2.]);

        assert!(a.argmax(0).equals(&b, 0.001));
    }
}
