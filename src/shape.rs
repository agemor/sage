use std;
use std::{slice, fmt};
use std::ops::{Index, IndexMut};
use std::fmt::{Display, Formatter};

const MAX_SHAPE_LEN: usize = 9;

// simple arrayvec
#[derive(Copy, Clone, Debug)]
pub struct Shape {
    sizes: [usize; MAX_SHAPE_LEN],
    len: usize,
}


#[derive(Debug, Clone)]
pub struct ShapeError {
    pub msg: String,
}

impl ShapeError {
    pub fn new(msg: &str) -> Self {
        ShapeError {
            msg: msg.to_string(),
        }
    }

    pub fn invalid_broadcast(a: Shape, b: Shape) -> Self {
        ShapeError::new(&*format!("invalid broadcast: {} and {} are not compatible", a, b))
    }
}


impl Shape {
    pub fn new(sizes: &[usize]) -> Self
    {
        if check_zero_sizes(sizes) {
            panic!("zero-length dimension is not allowed");
        }

        let mut shape = Shape::empty();
        shape.sizes[..sizes.len()].copy_from_slice(sizes);
        shape.len = sizes.len();

        shape
    }

    pub fn empty() -> Self {
        Shape { sizes: [1; MAX_SHAPE_LEN], len: 0 }
    }

    pub fn union<S, T>(shape_a: S, shape_b: T) -> Result<Self, ShapeError>
        where S: ToShape,
              T: ToShape
    {
        let shape_a = shape_a.to_shape();
        let shape_b = shape_b.to_shape();


        if shape_a == shape_b {
            Ok(shape_a)
        }

        // Do union
        else {
            let (longer, shorter) = if shape_a.len() > shape_b.len() { (shape_a, shape_b) } else { (shape_b, shape_a) };

            let mut u = shorter;

            for i in 0..(longer.len() - shorter.len()) {
                u.insert(i, longer[i]);
            }

            for (a, b) in u.iter_mut().zip(longer.iter()) {
                if *a != *b {
                    if *a == 1 {
                        *a = *b;
                    } else if *b != 1 {
                        return Err(ShapeError::new("invalid broadcast"));
                    }
                }
            }
            Ok(u)
        }
    }

    pub fn default_strides<S>(shape: S) -> Vec<usize>
        where S: ToShape
    {
        let mut strides = vec![1_usize];
        let mut cum_prod = 1;

        // tensor with (A, B, C) shape
        // have (B*C, C, 1) strides as a default.

        for dim_size in shape.to_shape().iter().skip(1).rev() {
            cum_prod *= dim_size;
            strides.push(cum_prod);
        }
        strides.reverse();
        strides
    }


    pub fn len(&self) -> usize {
        self.len
    }

    pub fn size(&self) -> usize {
        self.sizes.iter().product()
    }

    pub fn sizes(&self) -> &[usize] {
        &self.sizes[..self.len]
    }

    pub fn sizes_mut(&mut self) -> &mut [usize] {
        &mut self.sizes[..self.len]
    }

    pub fn iter(&self) -> slice::Iter<'_, usize> {
        self.sizes().iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<'_, usize> {
        self.sizes_mut().iter_mut()
    }

    pub fn insert<I>(&mut self, axis: I, size: usize) -> &mut Self
        where I: ToIndex
    {
        let axis = axis.to_index(self.len);
        if size == 0 {
            panic!("shape size cannot be zero");
        }

        if MAX_SHAPE_LEN <= axis {
            panic!("axis exceeds maximum shape len");
        }


        let mut i = self.len;
        while i > axis {
            self.sizes[i] = self.sizes[i - 1];
            i -= 1;
        }

        self.sizes[axis] = size;
        self.len += 1;
        self
    }

    pub fn remove<I>(&mut self, axis: I) -> &mut Self
        where I: ToIndex
    {
        let axis = axis.to_index(self.len);

        let mut i = axis;
        while i < self.len - 1 {
            self.sizes[i] = self.sizes[i + 1];
            i += 1;
        }
        self.len -= 1;
        self
    }

    pub fn swap<I, J>(&mut self, axis_a: I, axis_b: J) -> &mut Self
        where I: ToIndex,
              J: ToIndex
    {
        let axis_a = axis_a.to_index(self.len);
        let axis_b = axis_b.to_index(self.len);

        self.sizes.swap(axis_a, axis_b);
        self
    }

    pub fn replace<I>(&mut self, axis: I, size: usize) -> &mut Self {
        let axis = axis.to_index(self.len);
        if size == 0 {
            panic!("shape size cannot be zero");
        }
        self.sizes[axis] = size;
        self
    }

    pub fn split<I>(&self, axis: I) -> (Shape, Shape)
        where I: ToIndex
    {
        let axis = axis.to_index(self.len);

        let (sizes_a, sizes_b) = self.sizes().split_at(axis);
        (Shape::new(sizes_a), Shape::new(sizes_b))
    }

    pub fn extend<S>(&mut self, shape: S) -> &mut Self
        where S: ToShape
    {
        let shape = shape.to_shape();

        if self.len + shape.len >= MAX_SHAPE_LEN {
            panic!("exceeds maximum shape len")
        }

        for (i, &size) in shape.iter().enumerate() {
            self.sizes[self.len + i] = size;
        }
        self.len += shape.len;
        self
    }
}


fn check_zero_sizes(sizes: &[usize]) -> bool {
    sizes.iter().any(|d| { *d == 0 })
}


impl IntoIterator for Shape {
    type Item = usize;
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        Iter { shape: self, index: 0 }
    }
}

pub struct Iter {
    shape: Shape,
    index: usize,
}

impl Iterator for Iter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.shape.len {
            Some(self.shape[self.index])
        } else {
            None
        }
    }
}

pub trait ToShape {
    fn to_shape(&self) -> Shape;
}

impl ToShape for Shape {
    fn to_shape(&self) -> Shape {
        self.clone()
    }
}

impl ToShape for &Shape {
    fn to_shape(&self) -> Shape {
        (*self).clone()
    }
}


// array literal
impl<const C: usize> ToShape for [usize; C] {
    fn to_shape(&self) -> Shape {
        Shape::new(self)
    }
}

// slice
impl<'a> ToShape for &'a [usize] {
    fn to_shape(&self) -> Shape {
        Shape::new(self)
    }
}

// vec
impl ToShape for Vec<usize> {
    fn to_shape(&self) -> Shape {
        Shape::new(self)
    }
}


impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.sizes[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.sizes[index]
    }
}

impl Eq for Shape {}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.sizes() == other.sizes()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.len == 0 {
            write!(f, "[empty]")
        } else if self.len == 1 {
            write!(f, "[{}]", self.sizes[0])
        } else {
            write!(f, "[{}", self.sizes[0]);
            for s in self.sizes().iter().skip(1) {
                write!(f, ", {}", s);
            };
            write!(f, "]")
        }
    }
}


pub trait ToIndex {
    fn to_index(&self, bound: usize) -> usize;
}

impl ToIndex for usize {
    fn to_index(&self, bound: usize) -> usize {
        assert_index_bounds(*self as isize, bound);
        *self
    }
}

impl ToIndex for isize {
    fn to_index(&self, bound: usize) -> usize {
        assert_index_bounds(*self, bound);
        if *self >= 0 {
            *self as usize
        } else {
            (*self + bound as isize) as usize
        }
    }
}

impl ToIndex for i32 {
    fn to_index(&self, bound: usize) -> usize {
        assert_index_bounds(*self as isize, bound);
        if *self >= 0 {
            *self as usize
        } else {
            (*self + bound as i32) as usize
        }
    }
}

fn assert_index_bounds(index: isize, bound: usize) {
    let bound = bound as isize;
    let upper = bound - 1;
    let lower = -bound;

    if index > upper || index < lower {
        panic!(format!("index out of range: expected index in range of {}..{}, but {} is given.", lower, upper, index));
    }
}


fn main() {
    println!("Hello, world!");
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_len() {
        assert_eq!([2, 2, 2, 2, 2].to_shape().len(), 5_usize);
    }

    #[test]
    fn test_size() {
        assert_eq!([2, 2, 2, 2, 2].to_shape().size(), 2_usize.pow(5));
    }

    #[test]
    fn test_zero_dims() {
        assert_eq!(check_zero_sizes(&[1, 0, 2]), true);
        assert_eq!(check_zero_sizes(&[1, 1, 2]), false);
    }

    #[test]
    fn test_insert() {
        let mut a = [1, 2].to_shape();
        assert_eq!(*a.insert(0, 1), [1, 1, 2].to_shape());
        assert_eq!(*a.insert(0, 2), [2, 1, 1, 2].to_shape());
        assert_eq!(*a.insert(3, 4), [2, 1, 1, 4, 2].to_shape());
        assert_eq!(*a.insert(5, 5), [2, 1, 1, 4, 2, 5].to_shape());
    }

    #[test]
    fn test_remove() {
        let mut a = [9, 1, 2, 3, 4, 5].to_shape();
        assert_eq!(*a.remove(5), [9, 1, 2, 3, 4].to_shape());
        assert_eq!(*a.remove(0), [1, 2, 3, 4].to_shape());
    }

    #[test]
    fn test_swap() {
        let mut a = [1, 2].to_shape();
        assert_eq!(*a.swap(1, 0), [2, 1].to_shape());
    }

    #[test]
    fn test_replace() {
        let mut a = [1, 2].to_shape();
        assert_eq!(*a.replace(1, 1), [1, 1].to_shape());
    }

    #[test]
    fn test_split() {
        let orig = [9, 1, 2, 3, 4, 5].to_shape();
        let (a, b) = orig.split(0);
        assert_eq!(a, Shape::empty());
        assert_eq!(b, orig);

        let (a, b) = orig.split(6);
        assert_eq!(a, orig);
        assert_eq!(b, Shape::empty());
    }

    #[test]
    fn test_extend() {
        let orig = [9, 1, 2, 3, 4, 5].to_shape();
        let (mut a, b) = orig.split(3);
        assert_eq!(*a.extend(b), orig);
    }


    #[test]
    fn test_union() {
        let a = [3, 3, 3, 1, 1, 1, 3, 3, 3].to_shape();
        let b = [1, 1, 1, 3, 7, 3, 1, 1, 1].to_shape();

        assert_eq!(Shape::union(a, b).unwrap(), [3, 3, 3, 3, 7, 3, 3, 3, 3].to_shape());
    }

    #[test]
    fn test_default_strides() {
        assert_eq!(Shape::default_strides([2, 2, 2, 2, 2]), vec![16, 8, 4, 2, 1]);
    }
}
