use crate::tensor::Tensor;

use std::marker::PhantomData;


pub struct CoordIter {
    dim: Vec<usize>,
    // all inline vec
    strides: Vec<usize>,
    coord: Vec<usize>,
    done: bool,
}

impl CoordIter {
    pub fn new(dim: &[usize], strides: &[usize]) -> Self {
        CoordIter {
            dim: dim.to_vec(),
            strides: strides.to_vec(),
            // coord starts at (0,0,0,0)
            coord: vec![0; dim.len()],
            done: false,
        }
    }

    pub fn to_index(&self) -> usize {
        self.coord
            .iter()
            .zip(self.strides.iter())
            .fold(0, |a, (c, s)| {
                a + c * s
            })
    }

    fn next_coord(&mut self) {
        // update coord
        let mut axis = self.dim.len() - 1;

        while !self.done {
            if self.coord[axis] + 1 < self.dim[axis] {
                self.coord[axis] += 1;
                break;
            } else if axis > 0 {
                self.coord
                    .iter_mut()
                    .skip(axis)
                    .for_each(|e| *e = 0);
                axis -= 1;
            } else {
                self.done = true;
            }
        }
    }
}

impl Iterator for CoordIter {
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


pub struct AlongAxisIter<'a> {
    t: &'a Tensor,
    axis: usize,
    index: usize,
}

impl<'a> AlongAxisIter<'a> {
    pub fn new(t: &'a Tensor, axis: usize) -> Self {
        AlongAxisIter {
            t,
            axis,
            index: 0,
        }
    }
}

impl<'a> Iterator for AlongAxisIter<'a> {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.t.dim[self.axis] {
            self.index += 1;
            Some(self.t.index_axis((self.index - 1) as isize, self.axis as isize))
        } else {
            None
        }
    }
}


pub struct Iter<'a> {
    ptr: *const f32,
    offset: usize,
    coord_iter: CoordIter,
    phantom: PhantomData<&'a f32>,
}

impl<'a> Iter<'a> {
    pub fn new(ptr: *const f32, offset: usize, dim: &[usize], strides: &[usize]) -> Self {
        Iter {
            ptr,
            offset,
            coord_iter: CoordIter::new(dim, strides),
            phantom: PhantomData
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.coord_iter.next().map(|index| {
            unsafe {
                let elem = self.ptr.offset((index + self.offset) as isize);
                elem.as_ref().unwrap()
            }
        })
    }
}

pub struct IterMut<'a> {
    ptr: *mut f32,
    offset: usize,
    coord_iter: CoordIter,
    phantom: PhantomData<&'a f32>,
}

impl<'a> IterMut<'a> {
    pub fn new(ptr: *mut f32, offset: usize, dim: &[usize], strides: &[usize]) -> Self {
        IterMut {
            ptr,
            offset,
            coord_iter: CoordIter::new(dim, strides),
            phantom: PhantomData
        }
    }
}

impl<'a> Iterator for IterMut<'a> {
    type Item = &'a mut f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.coord_iter.next().map(|index| {
            unsafe {
                let elem = self.ptr.offset((index + self.offset) as isize);
                elem.as_mut().unwrap()
            }
        })
    }
}


// shape (un-broadcasted, ordered)
// stride (ordered)
// i => 0 ... shape.size()- 1 => (i0, i2, i3, i4, ... i_n)
//
// save all...
// calculate on-demand


