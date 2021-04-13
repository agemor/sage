use crate::ops::{BinaryIndexOperation, BinaryLogicOperation, BinaryOperation};
use crate::shape::{Axes, Axis};
use crate::tensor::{Element, Tensor};
use num_traits::{Float, NumOps};

impl<T> Tensor<T>
where
    T: Element + NumOps,
{
    pub fn sum<As>(&self, axes: As, retain_axis: bool) -> Tensor<T>
    where
        As: Axes,
    {
        self.reduce_op(BinaryOperation::Add, axes, retain_axis)
    }
}

impl<T> Tensor<T>
where
    T: Element + PartialOrd,
{
    pub fn max<As>(&self, axes: As, retain_axis: bool) -> Tensor<T>
    where
        As: Axes,
    {
        self.reduce_logic_op(BinaryLogicOperation::Max, axes, retain_axis)
    }

    pub fn min<As>(&self, axes: As, retain_axis: bool) -> Tensor<T>
    where
        As: Axes,
    {
        self.reduce_logic_op(BinaryLogicOperation::Min, axes, retain_axis)
    }

    pub fn argmax<A>(&self, axis: A, retain_axis: bool) -> Tensor<u32>
    where
        A: Axis,
    {
        self.reduce_index_op(BinaryIndexOperation::Max, axis, retain_axis)
    }

    pub fn argmin<A>(&self, axis: A, retain_axis: bool) -> Tensor<u32>
    where
        A: Axis,
    {
        self.reduce_index_op(BinaryIndexOperation::Min, axis, retain_axis)
    }
}

impl<T> Tensor<T>
where
    T: Element + Float,
{
    pub fn mean<As>(&self, axes: As, retain_axis: bool) -> Tensor<T>
    where
        As: Axes,
    {
        let axes = axes.to_vec(self.order()).unwrap();
        let size = axes.iter().map(|a| self.shape()[*a]).product::<usize>();

        self.sum(axes, retain_axis) / T::from(size).unwrap()
    }

    pub fn variance<As>(&self, axes: As, retain_axis: bool) -> Tensor<T>
    where
        As: Axes,
    {
        let axes = axes.to_vec(self.order()).unwrap();
        let size = axes.iter().map(|a| self.shape()[*a]).product::<usize>();

        (self - self.mean(&axes, true))
            .square()
            .sum(&axes, retain_axis)
            / T::from(size - 1).unwrap()
    }
}
