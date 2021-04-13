use crate::backend::{Backend, Buffer};
use crate::iter::{AlongAxisIter, IndexIter, Iter, Parallel};
use crate::ops::{BinaryIndexOperation, BinaryLogicOperation, BinaryOperation, UnaryOperation};
use crate::shape;
use crate::shape::{Axes, Axis, Shape};
use itertools::Itertools;
use num_traits::{Float, NumOps, One, Pow, WrappingNeg, Zero};
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::Normal;
use std::fmt;
use std::fmt::Debug;
use std::iter::Sum;
use std::sync::Arc;

pub trait Element: Copy + Send + Sync + Zero + One + Debug + 'static {}

pub struct Tensor<T> {
    // array offset and strides
    pub mem_layout: MemoryLayout,

    // underlying array that holds actual data
    pub buffer: Arc<Buffer<T>>,
}

impl<T> Tensor<T> {
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Properties
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub fn mem_layout(&self) -> &MemoryLayout {
        &self.mem_layout
    }

    pub fn order(&self) -> usize {
        self.mem_layout.num_axes()
    }

    pub fn shape(&self) -> &[usize] {
        self.mem_layout.extents()
    }

    pub fn strides(&self) -> &[usize] {
        self.mem_layout.strides()
    }

    pub fn offset(&self) -> usize {
        self.mem_layout.offset
    }

    pub fn size(&self) -> usize {
        self.mem_layout.size()
    }
}

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn backend(&self) -> Backend {
        self.buffer.backend()
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Basic Constructors
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub fn new<S>(shape: S, data: &[T], backend: Backend) -> Self
    where
        S: Shape,
    {
        Tensor::from_iter(shape, data.iter().copied(), backend)
    }

    pub fn from_buffer<S>(shape: S, buffer: Buffer<T>) -> Self
    where
        S: Shape,
    {
        let shape = shape.to_vec(buffer.size()).unwrap();

        Tensor {
            mem_layout: MemoryLayout::with_default(&shape),
            buffer: Arc::new(buffer),
        }
    }

    pub fn from_iter<S, I>(shape: S, data: I, backend: Backend) -> Self
    where
        S: Shape,
        I: ExactSizeIterator<Item = T>,
    {
        let shape = shape.to_vec(data.len()).unwrap();

        Tensor {
            mem_layout: MemoryLayout::with_default(&shape),
            buffer: Arc::new(backend.alloc_mem_from_iter(data)),
        }
    }

    pub fn from_elem<S>(shape: S, elem: T, backend: Backend) -> Self
    where
        S: Shape,
    {
        let shape = shape.to_vec(0).unwrap();
        let iter = (0..shape.iter().product()).map(|_| elem);

        Tensor::from_iter(shape, iter, backend)
    }

    pub fn from_dist<S, D>(shape: S, dist: D, backend: Backend) -> Self
    where
        S: Shape,
        D: Distribution<T>,
    {
        let shape = shape.to_vec(0).unwrap();

        let mut rng = thread_rng();
        let iter = (0..shape.iter().product()).map(|_| dist.sample(&mut rng));

        Tensor::from_iter(shape, iter, backend)
    }

    pub fn assign(&self, backend: Backend) -> Tensor<T> {
        Backend::assign(self, backend)
    }

    pub fn twin(&self) -> Tensor<T> {
        Tensor {
            mem_layout: self.mem_layout.clone(),
            buffer: self.buffer.clone(),
        }
    }

    fn with_mem_layout(&self, mem_layout: MemoryLayout) -> Tensor<T> {
        Tensor {
            mem_layout,
            buffer: self.buffer.clone(),
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Helper Constructors
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub fn zeros<S>(shape: S, backend: Backend) -> Self
    where
        S: Shape,
    {
        Tensor::from_elem(shape, T::zero(), backend)
    }

    pub fn ones<S>(shape: S, backend: Backend) -> Self
    where
        S: Shape,
    {
        Tensor::from_elem(shape, T::one(), backend)
    }

    // sample from standard normal dist
    pub fn randn<S>(shape: S, backend: Backend) -> Tensor<f32>
    where
        S: Shape,
    {
        Tensor::from_dist(
            shape,
            Normal::new(f32::zero(), f32::one()).unwrap(),
            backend,
        )
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Converters
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Warning: this operation moves data to CPU
    pub fn to_vec(&self) -> Vec<T> {
        self.buffer.to_vec()
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Iterators
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub fn iter(&self) -> Iter<T> {
        Iter::from_tensor(self)
    }

    pub fn par_iter(&self) -> Parallel<Iter<T>> {
        Parallel {
            iter: Iter::from_tensor(self),
        }
    }

    pub fn along_axis<A>(&self, axis: A) -> AlongAxisIter<T>
    where
        A: Axis,
    {
        let axis = axis.to_usize(self.order()).unwrap();
        AlongAxisIter::new(self, axis)
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Concat and stack
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub fn concat(inputs: &[&Tensor<T>], axis: usize) -> Tensor<T> {
        inputs[0].backend().concat(inputs, axis)
    }

    pub fn stack(inputs: &[&Tensor<T>], axis: usize) -> Tensor<T> {
        if !inputs.iter().map(|t| t.shape()).all_equal() {
            panic!("all tensors should be in the same shape");
        }

        let expanded_inputs = inputs.iter().map(|t| t.expand_axis(axis)).collect_vec();
        Self::concat(expanded_inputs.iter().collect_vec().as_ref(), axis)
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Memory Layout Helpers
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub fn squeeze_axis<A>(&self, axis: A) -> Tensor<T>
    where
        A: Axis,
    {
        let axis = axis.to_usize(self.order()).unwrap();

        if self.shape()[axis] != 1 {
            panic!("only size=1 axes can be squeezed");
        }

        let mut mem_layout = self.mem_layout.clone();
        mem_layout.remove_axis(axis);

        self.with_mem_layout(mem_layout)
    }

    pub fn expand_axis<A>(&self, axis: A) -> Tensor<T>
    where
        A: Axis,
    {
        // allow non-existing index
        let axis = axis.to_usize(self.order() + 1).unwrap();

        let mut mem_layout = self.mem_layout.clone();
        mem_layout.insert_axis(axis);

        self.with_mem_layout(mem_layout)
    }

    // reshape (underlying data does not change)
    pub fn reshape<S>(&self, shape: S) -> Tensor<T>
    where
        S: Shape,
    {
        let shape = shape.to_vec(self.size()).unwrap();
        let mem_layout = MemoryLayout::with_default(&shape);

        if self.mem_layout.is_contiguous() {
            self.with_mem_layout(mem_layout)
        } else {
            // create a new copy (with default memory layouts)
            self.clone().with_mem_layout(mem_layout)
        }
    }

    // swap last two dims of tensor
    pub fn transpose<A1, A2>(&self, axis1: A1, axis2: A2) -> Tensor<T>
    where
        A1: Axis,
        A2: Axis,
    {
        let axis1 = axis1.to_usize(self.order()).unwrap();
        let axis2 = axis2.to_usize(self.order()).unwrap();

        if axis1 == axis2 {
            panic!("same axis");
        }

        let mut mem_layout = self.mem_layout.clone();
        mem_layout.swap_axis(axis1, axis2);

        self.with_mem_layout(mem_layout)
    }

    pub fn permute<As>(&self, axes: As) -> Tensor<T>
    where
        As: Axes,
    {
        let axes = axes.to_vec(self.order()).unwrap();

        let mut use_counts = vec![0; self.order()];

        axes.iter().for_each(|axis| {
            use_counts[*axis] += 1;
        });

        if use_counts.iter().any(|count| *count != 1) {
            panic!("some axes are not used, or used more than once");
        }

        let mut mem_layout = self.mem_layout.clone();
        mem_layout.permute_axes(&axes);

        self.with_mem_layout(mem_layout)
    }

    fn broadcast<S>(&self, shape: S) -> Tensor<T>
    where
        S: Shape,
    {
        let shape = shape.to_vec(0).unwrap();

        let mem_layout = self.mem_layout.broadcast(&shape);

        self.with_mem_layout(mem_layout)
    }

    fn broadcast_do<F>(&self, other: &Tensor<T>, f: F) -> Tensor<T>
    where
        F: Fn(&Tensor<T>, &Tensor<T>) -> Tensor<T>,
    {
        let shape1 = self.shape();
        let shape2 = other.shape();

        if shape1 != shape2 {
            let union_shape = shape::union(shape1, shape2).unwrap();
            if shape1 == union_shape {
                f(self, &other.broadcast(union_shape))
            } else if shape2 == union_shape {
                f(&self.broadcast(union_shape), other)
            } else {
                f(&self.broadcast(&union_shape), &other.broadcast(union_shape))
            }
        } else {
            f(self, other)
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Indexing Operations
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub fn index<A>(&self, index: usize, axis: A) -> Tensor<T>
    where
        A: Axis,
    {
        let axis = axis.to_usize(self.order()).unwrap();
        let axis_size = self.shape()[axis];

        if axis_size <= index {
            panic!("index out of bounds");
        }

        let mut mem_layout = self.mem_layout.clone();
        mem_layout.select_index(index, axis);

        self.with_mem_layout(mem_layout)
    }

    pub fn slice<A>(&self, start_index: usize, end_index: usize, axis: A) -> Tensor<T>
    where
        A: Axis,
    {
        let axis = axis.to_usize(self.order()).unwrap();
        let axis_size = self.shape()[axis];

        if start_index > end_index {
            panic!("start and end index are not in the order");
        }

        if axis_size <= end_index {
            panic!("index out of bounds");
        }

        let mut mem_layout = self.mem_layout.clone();
        mem_layout.select_range(start_index, end_index, axis);

        self.with_mem_layout(mem_layout)
    }
}

impl<T> Tensor<T>
where
    T: Element + Float,
{
    pub fn unary_op(&self, op: UnaryOperation) -> Tensor<T> {
        self.backend().unary_op(self, op)
    }
}

impl<T> Tensor<T>
where
    T: Element + NumOps,
{
    pub fn binary_op(&self, other: &Tensor<T>, op: BinaryOperation) -> Tensor<T> {
        self.broadcast_do(other, move |t1, t2| self.backend().binary_op(t1, t2, op))
    }

    pub fn reduce_op<As>(&self, op: BinaryOperation, axes: As, retain_axis: bool) -> Tensor<T>
    where
        As: Axes,
    {
        let axes = axes.to_vec(self.order()).unwrap();

        self.backend().reduction_op(self, op, &axes)
    }
}

impl<T> Tensor<T>
where
    T: Element + PartialOrd,
{
    pub fn binary_logic_op(&self, other: &Tensor<T>, op: BinaryLogicOperation) -> Tensor<T> {
        self.broadcast_do(other, |t1, t2| self.backend().binary_logic_op(t1, t2, op))
    }

    pub fn reduce_logic_op<As>(
        &self,
        op: BinaryLogicOperation,
        axes: As,
        retain_axis: bool,
    ) -> Tensor<T>
    where
        As: Axes,
    {
        let axes = axes.to_vec(self.order()).unwrap();

        self.backend().reduction_logic_op(self, op, &axes)
    }

    pub fn reduce_index_op<A>(
        &self,
        op: BinaryIndexOperation,
        axis: A,
        retain_axis: bool,
    ) -> Tensor<u32>
    where
        A: Axis,
    {
        let axis = axis.to_usize(self.order()).unwrap();
        self.backend().reduction_index_op(self, op, axis)
    }
}

impl<T> Tensor<T>
where
    T: Element + NumOps + Sum,
{
    pub fn contract<As1, As2>(&self, other: &Tensor<T>, axes1: As1, axes2: As2) -> Tensor<T>
    where
        As1: Axes,
        As2: Axes,
    {
        let axes1 = axes1.to_vec(self.order()).unwrap();
        let axes2 = axes2.to_vec(other.order()).unwrap();

        self.backend().contraction(self, other, &axes1, &axes2)
    }
}

impl<T> Eq for Tensor<T> where T: Element + NumOps + PartialOrd {}

impl<T> PartialEq for Tensor<T>
where
    T: Element + NumOps + PartialOrd,
{
    fn eq(&self, other: &Self) -> bool {
        // check shape
        if self.shape() != other.shape() {
            return false;
        }

        // check inner contents
        let eq = self.binary_logic_op(other, BinaryLogicOperation::NotEq);

        todo!()
        //eq.sum((0..self.order()).collect_vec(), false).to_vec()[0] == T::zero()
    }
}

impl<T> Clone for Tensor<T>
where
    T: Element,
{
    fn clone(&self) -> Self {
        self.backend().copy(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryLayout {
    pub extents: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl MemoryLayout {
    pub fn default_strides(extents: &[usize]) -> Vec<usize> {
        let size = extents.iter().product();
        extents
            .iter()
            .scan(size, |size, extent| {
                *size /= extent;
                Some(*size)
            })
            .collect()
    }

    pub fn new(extents: &[usize], strides: &[usize], offset: usize) -> MemoryLayout {
        MemoryLayout {
            extents: extents.to_vec(),
            strides: strides.to_vec(),
            offset,
        }
    }

    pub fn with_default(extents: &[usize]) -> MemoryLayout {
        MemoryLayout {
            extents: extents.to_vec(),
            strides: Self::default_strides(&extents),
            offset: 0,
        }
    }

    pub fn num_axes(&self) -> usize {
        self.extents.len()
    }

    pub fn extents(&self) -> &[usize] {
        &self.extents
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn size(&self) -> usize {
        self.extents.iter().product()
    }

    pub fn translate_default(&self, index: usize) -> usize {
        let mut out_index = self.offset;
        let mut p = self.size();
        let mut rem = index;
        for i in 0..self.extents.len() {
            p /= self.extents[i];
            let c = rem / p;
            rem -= c * p;
            out_index += c * self.strides[i];
        }
        out_index
    }

    pub fn translate(&self, index: usize, reference: &MemoryLayout) -> usize {
        let mut out_index = self.offset;
        let mut rem = index - reference.offset;
        for i in 0..self.extents.len() {
            let c = rem / reference.strides[i];
            rem -= c * reference.strides[i];
            out_index += c * self.strides[i];
        }
        out_index
    }

    pub fn convert_coord(&self, index: usize) {}

    pub fn split(&self, axes: &[usize], nested: bool) -> (MemoryLayout, MemoryLayout) {
        let not_axes = (0..self.extents.len()).filter(|a| !axes.contains(a));

        let left = MemoryLayout {
            extents: axes.iter().map(|&a| self.extents[a]).collect(),
            strides: axes.iter().map(|&a| self.strides[a]).collect(),
            offset: self.offset,
        };

        let right = MemoryLayout {
            extents: not_axes.clone().map(|a| self.extents[a]).collect(),
            strides: not_axes.map(|a| self.strides[a]).collect(),
            offset: if nested { 0 } else { self.offset },
        };

        (left, right)
    }

    // Whether the indices described by this memory layout is contiguous
    // Returns true if there are no indexing (or slicing) operations.
    // This method asks: Did user do any indexing (slicing) operations?
    pub fn is_contiguous(&self) -> bool {
        // max index == (unbroadcasted) theoretical max index
        let max_index = self
            .extents
            .iter()
            .zip(self.strides.iter())
            .map(|(extent, stride)| (extent - 1) * stride)
            .sum::<usize>();

        let t_max_index = self
            .extents
            .iter()
            .zip(self.strides.iter())
            .filter(|(_, &stride)| stride > 0)
            .map(|(&extent, _)| extent)
            .product::<usize>()
            - 1;

        max_index == t_max_index
    }

    // one-to-one correspondence
    // This method asks: Did user do any broadcasting operations?
    pub fn is_bijective(&self) -> bool {
        self.strides.iter().all(|&stride| stride > 0)
    }

    // This method asks: did the user used any reshape operations?
    pub fn is_ordered(&self) -> bool {
        self.strides
            .iter()
            .filter(|&stride| *stride > 0)
            .is_sorted_by(|&a, &b| Some(b.cmp(a)))
    }

    pub fn remove_axis(&mut self, axis: usize) {
        self.extents.remove(axis);
        self.strides.remove(axis);
    }

    pub fn insert_axis(&mut self, axis: usize) {
        self.extents.insert(axis, 1);
        self.strides.insert(axis, 0);
    }

    pub fn swap_axis(&mut self, axis1: usize, axis2: usize) {
        self.extents.swap(axis1, axis2);
        self.strides.swap(axis1, axis2);
    }

    pub fn permute_axes(&mut self, axes: &[usize]) {
        let (new_extents, new_strides) = axes
            .iter()
            .map(|axis| (self.extents[*axis], self.strides[*axis]))
            .unzip();
        self.extents = new_extents;
        self.strides = new_strides;
    }

    pub fn select_index(&mut self, index: usize, axis: usize) {
        if self.num_axes() <= axis {
            panic!("axis out of bounds");
        }

        if self.extents[axis] <= index {
            panic!("index out of bounds");
        }

        self.offset += self.strides[axis] * index;
        self.remove_axis(axis);
    }

    pub fn select_range(&mut self, index_start: usize, index_end: usize, axis: usize) {
        if self.extents[axis] <= index_end {
            panic!("index out of bounds");
        }

        self.offset += self.strides[axis] * index_start;
        self.extents[axis] = index_end - index_start + 1;
        self.strides.remove(axis);
    }

    pub fn broadcast(&self, extents: &[usize]) -> MemoryLayout {
        if extents.len() < self.extents.len() {
            panic!("target shape must be larger than the broadcasted shape");
        }

        let mut new_extents = self.extents.clone();
        let mut new_strides = self.strides.clone();

        // Let's say that we are broadcasting
        // (3, 1, 5) to (2, 1, 3, 9, 5)

        // First, we add padding so that
        // (3, 1, 5) -> (1, 1, 3, 1, 5)
        for _ in 0..(extents.len() - self.extents.len()) {
            new_extents.insert(0, 1);
            new_strides.insert(0, 0);
        }

        // Next, we update extents while checking its validity
        for ((new_extent, extent), new_stride) in new_extents
            .iter_mut()
            .zip(extents.iter())
            .zip(new_strides.iter_mut())
        {
            if *new_extent != *extent {
                // for broadcasted axes, 'mute' them by set its stride to zero
                if *new_extent == 1 {
                    *new_extent = *extent;
                    *new_stride = 0;
                } else {
                    panic!("invalid broadcast... target shape should be larger.");
                }
            }
        }

        MemoryLayout {
            extents: new_extents,
            strides: new_strides,
            offset: self.offset,
        }
    }

    pub fn iter(&self) -> IndexIter {
        IndexIter::new(&self)
    }

    pub fn par_iter(&self) -> Parallel<IndexIter> {
        Parallel {
            iter: IndexIter::new(&self),
        }
    }
}
impl Element for isize {}
impl Element for i32 {}
impl Element for i16 {}
impl Element for i8 {}

impl Element for usize {}
impl Element for u32 {}
impl Element for u16 {}
impl Element for u8 {}

//impl Element for f64 {}
impl Element for f32 {}

#[cfg(test)]
mod tests {
    use crate::backend::Backend;
    use crate::tensor::Tensor;

    #[test]
    fn test_concat() {
        let t1 = Tensor::from_elem([3, 1, 2], 1, Backend::Native);
        let t2 = Tensor::from_elem([3, 3, 2], 3, Backend::Native);

        let c = Tensor::concat(&[&t1, &t2], 1);
    }
}
