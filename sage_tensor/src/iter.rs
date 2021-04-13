use crate::tensor::{Element, MemoryLayout, Tensor};
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

pub struct IndexIter {
    mem_layout: MemoryLayout,
    mem_layout_default: MemoryLayout,

    index: usize,
    len: usize,
}

impl IndexIter {
    pub fn new(mem_layout: &MemoryLayout) -> Self {
        IndexIter {
            mem_layout: mem_layout.clone(),
            mem_layout_default: MemoryLayout::with_default(&mem_layout.extents),
            index: 0,
            len: mem_layout.size(),
        }
    }

    pub fn from_tensor<T>(tensor: &Tensor<T>) -> Self {
        Self::new(&tensor.mem_layout)
    }
}

impl Iterator for IndexIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let t_index = self
                .mem_layout
                .translate(self.index, &self.mem_layout_default);
            self.index += 1;
            Some(t_index)
        } else {
            None
        }
    }
}

impl DoubleEndedIterator for IndexIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let t_index = self
                .mem_layout
                .translate(self.len - 1, &self.mem_layout_default);
            self.len -= 1;
            Some(t_index)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for IndexIter {
    fn len(&self) -> usize {
        self.len
    }
}

impl Producer for IndexIter {
    type Item = <IndexIter as Iterator>::Item;
    type IntoIter = IndexIter;

    fn into_iter(self) -> Self::IntoIter {
        self
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left = IndexIter {
            mem_layout: self.mem_layout.clone(),
            mem_layout_default: self.mem_layout_default.clone(),
            index: self.index,
            len: self.index + index,
        };
        let right = IndexIter {
            mem_layout: self.mem_layout.clone(),
            mem_layout_default: self.mem_layout_default.clone(),
            index: self.index + index,
            len: self.len,
        };
        (left, right)
    }
}

pub struct Iter<'a, T> {
    buffer: &'a [T],
    index_iter: IndexIter,
}

impl<'a, T> Iter<'a, T>
where
    T: Element,
{
    pub fn from_tensor(tensor: &Tensor<T>) -> Self {
        let a = tensor.buffer.as_native().data();
        Iter {
            buffer: a.as_slice(),
            index_iter: IndexIter::from_tensor(tensor),
        }
    }
}

impl<T> Iterator for Iter<'_, T>
where
    T: Element,
{
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iter.next().map(|i| (i, self.buffer[i]))
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T>
where
    T: Element,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.index_iter.next_back().map(|i| (i, self.buffer[i]))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T>
where
    T: Element,
{
    fn len(&self) -> usize {
        self.index_iter.len()
    }
}

impl<T> Producer for Iter<'_, T>
where
    T: Element,
{
    type Item = <Self as Iterator>::Item;
    type IntoIter = Self;

    fn into_iter(self) -> Self::IntoIter {
        self
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_index_iter, right_index_iter) = self.index_iter.split_at(index);

        let left = Iter {
            buffer: self.buffer,
            index_iter: left_index_iter,
        };

        let right = Iter {
            buffer: self.buffer,
            index_iter: right_index_iter,
        };

        (left, right)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Parallel<I> {
    pub iter: I,
}

impl<I> ParallelIterator for Parallel<I>
where
    I: Iterator + DoubleEndedIterator + ExactSizeIterator + Producer,
    <I as Producer>::Item: Send + Sync,
{
    type Item = <I as Producer>::Item;

    fn drive_unindexed<C>(self, consumer: C) -> <C as Consumer<Self::Item>>::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

impl<I> IndexedParallelIterator for Parallel<I>
where
    I: Iterator + DoubleEndedIterator + ExactSizeIterator + Producer,
    <I as Producer>::Item: Send + Sync,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> <C as Consumer<Self::Item>>::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> <CB as ProducerCallback<Self::Item>>::Output {
        callback.callback(self.iter)
    }
}

pub struct AlongAxisIter<'a, T> {
    t: &'a Tensor<T>,
    axis: usize,
    index: usize,
}

impl<'a, T> AlongAxisIter<'a, T> {
    pub fn new(t: &'a Tensor<T>, axis: usize) -> Self {
        AlongAxisIter { t, axis, index: 0 }
    }
}

impl<'a, T> Iterator for AlongAxisIter<'a, T>
where
    T: Element,
{
    type Item = Tensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.t.shape()[self.axis] {
            self.index += 1;
            Some(self.t.index(self.index - 1, self.axis))
        } else {
            None
        }
    }
}
