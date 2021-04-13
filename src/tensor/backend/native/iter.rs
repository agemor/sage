use crate::tensor::Tensor;
use std::cell::{Ref, RefMut};
use std::marker::PhantomData;

pub struct BufferIndexIter {
    extents: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    coord: Vec<usize>,
    done: bool,
}

impl BufferIndexIter {
    pub fn new(extents: &[usize], strides: &[usize], offset: usize) -> Self {
        BufferIndexIter {
            extents: extents.to_vec(),
            strides: strides.to_vec(),
            offset,
            // coord starts at (0,0,0,0)
            coord: vec![0; extents.len()],
            done: false,
        }
    }

    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self::new(tensor.shape.sizes(), tensor.strides(), tensor.offset)
    }

    pub fn to_index(&self) -> usize {
        self.offset
            + self
                .coord
                .iter()
                .zip(self.strides.iter())
                .fold(0, |a, (c, s)| a + c * s)
    }

    fn next_coord(&mut self) {
        // update coord
        let mut axis = self.extents.len() - 1;

        while !self.done {
            if self.coord[axis] + 1 < self.extents[axis] {
                self.coord[axis] += 1;
                break;
            } else if axis > 0 {
                self.coord.iter_mut().skip(axis).for_each(|e| *e = 0);
                axis -= 1;
            } else {
                self.done = true;
            }
        }
    }
}

impl Iterator for BufferIndexIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let index = self.to_index();
        self.next_coord();
        Some(index)
    }
}

pub struct Iter<'a> {
    data: Ref<'a, Vec<f32>>,
    coord_iter: BufferIndexIter,
    phantom: PhantomData<&'a f32>,
}

impl<'a> Iter<'a> {
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Iter {
            data: tensor.buffer.as_native().data.borrow(),
            coord_iter: BufferIndexIter::from_tensor(tensor),
            phantom: PhantomData,
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = (usize, &'a f32);

    fn next(&mut self) -> Option<Self::Item> {
        self.coord_iter
            .next()
            .map(|index| (index, &self.data[index]))
    }
}

pub struct IterMut<'a> {
    data: RefMut<'a, Vec<f32>>,
    coord_iter: BufferIndexIter,
    phantom: PhantomData<&'a f32>,
}

impl<'a> IterMut<'a> {
    pub fn from_tensor(tensor: &Tensor) -> Self {
        IterMut {
            data: tensor.buffer.as_native().data.borrow_mut(),
            coord_iter: BufferIndexIter::from_tensor(tensor),
            phantom: PhantomData,
        }
    }
}

impl<'a> Iterator for IterMut<'a> {
    type Item = (usize, &'a mut f32);

    fn next(&mut self) -> Option<Self::Item> {
        self.coord_iter
            .next()
            .map(|index| (index, &mut self.data[index]))
    }
}
