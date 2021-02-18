use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};


#[derive(Debug, Clone)]
pub struct ShapeError {
    pub msg: String,
}

impl ShapeError {
    pub(crate) fn new(msg: &str) -> Self {
        ShapeError {
            msg: msg.to_string(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Dim {
    pub sizes: Vec<usize>
}

impl Dim {
    pub fn new(sizes: &[usize]) -> Self {
        if check_zero_dims(sizes) {
            panic!("zero-length dimension is not allowed");
        }
        Dim { sizes: sizes.to_vec() }
    }

    pub fn empty() -> Self {
        Dim::new(&[])
    }

    pub fn from<D>(shape: D) -> Self
        where D: IntoDimension
    {
        shape.into_dimension()
    }

    pub fn union_k<D>(a:D, b:D) -> Result<Dim, ShapeError>
    where D: IntoDimension
    {

        let a= a.into_dimension();
        let b= b.into_dimension();

        if a == b {
            Ok(a)
        }
        // Do union
        else {
            let (longer, shorter) = if a.ndim() > b.ndim() { (a, b) } else { (b, a) };

            let mut union_dim = shorter.clone();

            for i in 0..(longer.ndim() - shorter.ndim()) {
                union_dim.add(i, longer[i]);
            }

            for (a, b) in union_dim.iter_mut().zip(longer.iter()) {
                if *a != *b {
                    if *a == 1 {
                        *a = *b;
                    } else if *b != 1 {
                        return Err(ShapeError::new("invalid broadcast"));
                    }
                }
            }
            Ok(union_dim)
        }
    }



    pub fn size(&self) -> usize {
        self.sizes.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.sizes.len()
    }

    pub fn union(&self, other: &Dim) -> Result<Dim, ShapeError> {
        if self == other {
            Ok(self.clone())
        }
        // Do union
        else {
            let (longer, shorter) = if self.ndim() > other.ndim() { (self, other) } else { (other, self) };

            let mut union_dim = shorter.clone();

            for i in 0..(longer.ndim() - shorter.ndim()) {
                union_dim.add(i, longer[i]);
            }

            for (a, b) in union_dim.iter_mut().zip(longer.iter()) {
                if *a != *b {
                    if *a == 1 {
                        *a = *b;
                    } else if *b != 1 {
                        return Err(ShapeError::new("invalid broadcast"));
                    }
                }
            }
            Ok(union_dim)
        }
    }

    // calculate default strides for tensor with this dim
    pub fn default_strides(&self) -> Vec<usize> {
        let mut strides = vec![1_usize];
        let mut cum_prod = 1;

        // tensor with (A, B, C) shape
        // have (B*C, C, 1) strides as a default.

        for dim_size in self.iter().skip(1).rev() {
            cum_prod *= dim_size;
            strides.push(cum_prod);
        }
        strides.reverse();
        strides
    }

    pub fn iter(&self) -> Iter<usize> {
        self.sizes.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<usize> {
        self.sizes.iter_mut()
    }

    pub fn add(&mut self, index: usize, size: usize) -> &mut Self {
        if size == 0 {
            panic!("dimension size cannot be zero");
        }

        if self.ndim() <= index {
            self.sizes.push(size);
        } else {
            self.sizes.insert(index, size);
        }
        self
    }

    pub fn remove(&mut self, index: usize) -> &mut Self {
        self.sizes.remove(index);
        self
    }

    pub fn split(&self, axis: usize) -> (Dim, Dim) {
        let (sizes_a, sizes_b) = self.sizes.split_at(axis);
        (Dim::new(sizes_a), Dim::new(sizes_b))
    }

    pub fn extend(&mut self, other: &Dim) -> &mut Self {
        self.sizes.extend(&other.sizes);
        self
    }
}


pub trait IntoDimension {
    fn into_dimension(self) -> Dim;
}

// array literal
impl<const C: usize> IntoDimension for [usize; C] {
    fn into_dimension(self) -> Dim {
        Dim::new(&self)
    }
}

// slice
impl<'a> IntoDimension for &'a [usize] {
    fn into_dimension(self) -> Dim {
        Dim::new(self)
    }
}

// vec
impl IntoDimension for Vec<usize> {
    fn into_dimension(self) -> Dim {
        Dim::new(&self)
    }
}

// Dim itself
impl IntoDimension for Dim {
    fn into_dimension(self) -> Dim {
        self
    }
}

// Dim itself
impl IntoDimension for &Dim {
    fn into_dimension(self) -> Dim {
        self.clone()
    }
}


impl Index<usize> for Dim {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.sizes[index]
    }
}

impl IndexMut<usize> for Dim {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.sizes[index]
    }
}


fn check_zero_dims(dim: &[usize]) -> bool {
    dim.iter().any(|d| { *d == 0 })
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndim() {
        assert_eq!(Dim::from([2, 2, 2, 2, 2]).ndim(), 5_usize);
    }

    #[test]
    fn test_size() {
        assert_eq!(Dim::from([2, 2, 2, 2, 2]).size(), 2_usize.pow(5));
    }

    #[test]
    fn test_zero_dims() {
        assert_eq!(check_zero_dims(&[1, 0, 2]), true);
        assert_eq!(check_zero_dims(&[1, 1, 2]), false);
    }

    #[test]
    fn test_add() {
        let mut a = Dim::from([1, 2]);
        assert_eq!(a.add(0, 1).clone(), Dim::from([1, 1, 2]));
        assert_eq!(a.add(0, 2).clone(), Dim::from([2, 1, 1, 2]));
        assert_eq!(a.add(3, 4).clone(), Dim::from([2, 1, 1, 4, 2]));
        assert_eq!(a.add(5, 5).clone(), Dim::from([2, 1, 1, 4, 2, 5]));
    }

    #[test]
    fn test_remove() {
        let mut a = Dim::from([9, 1, 2, 3, 4, 5]);
        assert_eq!(a.remove(5).clone(), Dim::from([9, 1, 2, 3, 4]));
        assert_eq!(a.remove(0).clone(), Dim::from([1, 2, 3, 4]));
    }

    #[test]
    fn test_split() {
        let orig = Dim::from([9, 1, 2, 3, 4, 5]);
        let (a, b) = orig.split(0);
        assert_eq!(a, Dim::from([]));
        assert_eq!(b, orig);

        let (a, b) = orig.split(6);
        assert_eq!(a, orig);
        assert_eq!(b, Dim::from([]));
    }

    #[test]
    fn test_extend() {
        let orig = Dim::from([9, 1, 2, 3, 4, 5]);
        let (mut a, b) = orig.split(3);
        assert_eq!(a.extend(&b).clone(), orig);
    }


    #[test]
    fn test_union() {
        let a = Dim::from([3, 3, 3, 1, 1, 1, 3, 3, 3]);
        let b = Dim::from([1, 1, 1, 3, 7, 3, 1, 1, 1]);

        assert_eq!(a.union(&b).unwrap(), Dim::from([3, 3, 3, 3, 7, 3, 3, 3, 3]));
    }

    #[test]
    fn test_default_strides() {
        assert_eq!(Dim::from([2, 2, 2, 2, 2]).default_strides(), vec![16, 8, 4, 2, 1]);
    }
}
