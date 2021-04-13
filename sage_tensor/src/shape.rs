use itertools::Itertools;
use num_integer::Integer;
use std::convert::TryInto;
use thiserror::Error;

#[derive(Error, Debug, Eq, PartialEq)]
pub enum ShapeError {
    #[error("size mismatch! expected {} but got {}.", .0, .1)]
    SizeMismatch(usize, usize),

    #[error("cannot infer the size")]
    InvalidInference,

    #[error("invalid shape extent {}, size should be larger than 0 or set to -1 for inference", .0)]
    InvalidExtent(isize),

    #[error("index out of range, expected index in range of {}..{}, but {} is given.", .low, .high, .index)]
    OutOfBounds {
        index: isize,
        low: isize,
        high: isize,
    },

    #[error("invalid index bound")]
    InvalidBound,

    #[error("invalid broadcast")]
    InvalidBroadcast,
}

pub trait Extent {
    fn needs_infer(&self) -> bool;
    fn to_usize(&self) -> Result<usize, ShapeError>;
}

pub trait Axis {
    fn to_usize(&self, bound: usize) -> Result<usize, ShapeError>;
}

pub trait Shape {
    fn to_vec(&self, size: usize) -> Result<Vec<usize>, ShapeError>;
}

pub trait Axes {
    fn to_vec(&self, bound: usize) -> Result<Vec<usize>, ShapeError>;
}

pub fn union<S1, S2>(shape1: S1, shape2: S2) -> Result<Vec<usize>, ShapeError>
where
    S1: Shape,
    S2: Shape,
{
    let shape1 = shape1.to_vec(0).unwrap();
    let shape2 = shape2.to_vec(0).unwrap();

    if shape1 == shape2 {
        Ok(shape1)
    }
    // Do union
    else {
        let (longer, shorter) = if shape1.len() > shape2.len() {
            (shape1, shape2)
        } else {
            (shape2, shape1)
        };

        let len = longer.len() - shorter.len();
        let mut u = shorter;

        for i in 0..len {
            u.insert(i, longer[i]);
        }

        for (a, b) in u.iter_mut().zip(longer.iter()) {
            if *a != *b {
                if *a == 1 {
                    *a = *b;
                } else if *b != 1 {
                    return Err(ShapeError::InvalidBroadcast);
                }
            }
        }
        Ok(u)
    }
}

fn axes_to_vec<A>(axes: &[A], bound: usize) -> Result<Vec<usize>, ShapeError>
where
    A: Axis,
{
    axes.iter().map(|i| i.to_usize(bound)).try_collect()
}

fn shape_to_vec<E>(extents: &[E], size: usize) -> Result<Vec<usize>, ShapeError>
where
    E: Extent,
{
    let mut use_infer = false;
    let mut infer_idx = 0;

    let mut expected_size = 1;
    let mut vec = Vec::with_capacity(extents.len());

    for (i, extent) in extents.iter().enumerate() {
        if extent.needs_infer() {
            if !use_infer {
                use_infer = true;
                infer_idx = i;
            } else {
                return Err(ShapeError::InvalidInference);
            }
        } else {
            let e = extent.to_usize()?;
            vec.push(e);
            expected_size *= e;
        }
    }

    if !use_infer && expected_size != size && size > 0 {
        return Err(ShapeError::SizeMismatch(size, expected_size));
    }

    if use_infer && size == 0 {
        return Err(ShapeError::InvalidInference);
    }

    if use_infer && !size.divides(&expected_size) {
        return Err(ShapeError::InvalidInference);
    }
    if use_infer {
        vec.insert(infer_idx, size / expected_size)
    }
    Ok(vec)
}

macro_rules! impl_extent_unsigned {
    ($ty:ty) => {
        impl Extent for $ty {
            fn needs_infer(&self) -> bool {
                false
            }

            fn to_usize(&self) -> Result<usize, ShapeError> {
                if *self > 0 {
                    Ok(*self as usize)
                } else {
                    Err(ShapeError::InvalidExtent(*self as isize))
                }
            }
        }
    };
}

macro_rules! impl_extent_signed {
    ($ty:ty) => {
        impl Extent for $ty {
            fn needs_infer(&self) -> bool {
                *self == -1
            }

            fn to_usize(&self) -> Result<usize, ShapeError> {
                if *self > 0 {
                    Ok(*self as usize)
                } else {
                    Err(ShapeError::InvalidExtent((*self).try_into().unwrap()))
                }
            }
        }
    };
}

macro_rules! impl_axis_unsigned {
    ($ty:ty) => {
        impl Axis for $ty {
            fn to_usize(&self, bound: usize) -> Result<usize, ShapeError> {
                if bound < 1 {
                    return Err(ShapeError::InvalidBound);
                }
                let axis = *self as usize;
                if axis < bound {
                    Ok(axis)
                } else {
                    Err(ShapeError::OutOfBounds {
                        index: axis as isize,
                        low: -(bound as isize),
                        high: (bound - 1) as isize,
                    })
                }
            }
        }
    };
}

macro_rules! impl_axis_signed {
    ($ty:ty) => {
        impl Axis for $ty {
            fn to_usize(&self, bound: usize) -> Result<usize, ShapeError> {
                if bound < 1 {
                    return Err(ShapeError::InvalidBound);
                }
                let axis = *self as isize;
                let axis = if axis >= 0 {
                    axis
                } else {
                    axis + bound as isize
                } as usize;

                if axis < bound {
                    Ok(axis)
                } else {
                    Err(ShapeError::OutOfBounds {
                        index: *self as isize,
                        low: -(bound as isize),
                        high: (bound - 1) as isize,
                    })
                }
            }
        }
    };
}

impl_extent_unsigned!(u8);
impl_extent_unsigned!(u16);
impl_extent_unsigned!(u32);
impl_extent_unsigned!(usize);

impl_extent_signed!(i8);
impl_extent_signed!(i16);
impl_extent_signed!(i32);
impl_extent_signed!(isize);

impl_axis_unsigned!(u8);
impl_axis_unsigned!(u16);
impl_axis_unsigned!(u32);
impl_axis_unsigned!(usize);

impl_axis_signed!(i8);
impl_axis_signed!(i16);
impl_axis_signed!(i32);
impl_axis_signed!(isize);

impl<T, const C: usize> Shape for [T; C]
where
    T: Extent,
{
    fn to_vec(&self, size: usize) -> Result<Vec<usize>, ShapeError> {
        shape_to_vec(self, size)
    }
}

impl<'a, T> Shape for &'a [T]
where
    T: Extent,
{
    fn to_vec(&self, size: usize) -> Result<Vec<usize>, ShapeError> {
        shape_to_vec(self, size)
    }
}

impl<T> Shape for Vec<T>
where
    T: Extent,
{
    fn to_vec(&self, size: usize) -> Result<Vec<usize>, ShapeError> {
        shape_to_vec(self, size)
    }
}

impl<T> Shape for &Vec<T>
where
    T: Extent,
{
    fn to_vec(&self, size: usize) -> Result<Vec<usize>, ShapeError> {
        shape_to_vec(self, size)
    }
}

impl<T, const C: usize> Axes for [T; C]
where
    T: Axis,
{
    fn to_vec(&self, bound: usize) -> Result<Vec<usize>, ShapeError> {
        axes_to_vec(self, bound)
    }
}

impl<'a, T> Axes for &'a [T]
where
    T: Axis,
{
    fn to_vec(&self, bound: usize) -> Result<Vec<usize>, ShapeError> {
        axes_to_vec(self, bound)
    }
}

impl<T> Axes for Vec<T>
where
    T: Axis,
{
    fn to_vec(&self, bound: usize) -> Result<Vec<usize>, ShapeError> {
        axes_to_vec(self, bound)
    }
}

impl<T> Axes for &Vec<T>
where
    T: Axis,
{
    fn to_vec(&self, bound: usize) -> Result<Vec<usize>, ShapeError> {
        axes_to_vec(self, bound)
    }
}

#[cfg(test)]
mod tests {
    use crate::shape::{axes_to_vec, shape_to_vec, union, Axis, Extent, ShapeError};

    fn axis_to_usize<A: Axis>(a: A, bound: usize) -> Result<usize, ShapeError> {
        a.to_usize(bound)
    }

    fn extent_to_usize<E: Extent>(e: E) -> Result<usize, ShapeError> {
        e.to_usize()
    }

    fn extent_needs_infer<E: Extent>(e: E) -> bool {
        e.needs_infer()
    }

    #[test]
    fn test_axis() {
        assert_eq!(axis_to_usize(-3_isize, 3).unwrap(), 0);
        assert_eq!(axis_to_usize(-2_i32, 3).unwrap(), 1);
        assert_eq!(axis_to_usize(-2_i16, 3).unwrap(), 1);
        assert_eq!(axis_to_usize(-1_i8, 3).unwrap(), 2);
        assert_eq!(axis_to_usize(0_usize, 3).unwrap(), 0);
        assert_eq!(axis_to_usize(1_u32, 3).unwrap(), 1);
        assert_eq!(axis_to_usize(2_u16, 3).unwrap(), 2);
        assert_eq!(axis_to_usize(2_u8, 3).unwrap(), 2);
    }

    #[test]
    fn test_axis_err_oob() {
        assert_eq!(axis_to_usize(0, 0).expect_err(""), ShapeError::InvalidBound);
        assert_eq!(
            axis_to_usize(-4, 3).expect_err(""),
            ShapeError::OutOfBounds {
                index: -4,
                low: -3,
                high: 2
            }
        );
        assert_eq!(
            axis_to_usize(4, 4).expect_err(""),
            ShapeError::OutOfBounds {
                index: 4,
                low: -4,
                high: 3
            }
        );
    }

    #[test]
    fn test_extent() {
        assert_eq!(extent_to_usize(1_usize).unwrap(), 1);
        assert_eq!(extent_to_usize(1_u32).unwrap(), 1);
        assert_eq!(extent_to_usize(1_u16).unwrap(), 1);
        assert_eq!(extent_to_usize(1_u8).unwrap(), 1);
        assert_eq!(extent_to_usize(1_isize).unwrap(), 1);
        assert_eq!(extent_to_usize(1_i32).unwrap(), 1);
        assert_eq!(extent_to_usize(1_i16).unwrap(), 1);
        assert_eq!(extent_to_usize(1_i8).unwrap(), 1);
    }

    #[test]
    fn test_extent_err_invalid() {
        assert_eq!(
            extent_to_usize(-1).expect_err(""),
            ShapeError::InvalidExtent(-1)
        );
        assert_eq!(
            extent_to_usize(0).expect_err(""),
            ShapeError::InvalidExtent(0)
        );
    }

    #[test]
    fn test_extent_needs_infer() {
        assert_eq!(extent_needs_infer(-2), false);
        assert_eq!(extent_needs_infer(-1), true);
        assert_eq!(extent_needs_infer(0), false);
        assert_eq!(extent_needs_infer(1), false);
    }

    #[test]
    fn test_axes_to_vec() {
        assert_eq!(
            axes_to_vec(&[-3, -2, -1, 0, 1, 2], 3).unwrap(),
            vec![0, 1, 2, 0, 1, 2]
        );
    }

    #[test]
    fn test_axes_to_vec_err() {
        assert_eq!(
            axes_to_vec(&[-4, -2, -1, 0, 1, 2], 3).expect_err(""),
            ShapeError::OutOfBounds {
                index: -4,
                low: -3,
                high: 2
            }
        );
    }

    #[test]
    fn test_shape_to_vec() {
        assert_eq!(shape_to_vec(&[2, 3, 4], 0).unwrap(), vec![2, 3, 4]);
        assert_eq!(shape_to_vec(&[2, 3, 4], 24).unwrap(), vec![2, 3, 4]);
        assert_eq!(shape_to_vec(&[-1, 3, 4], 24).unwrap(), vec![2, 3, 4]);
        assert_eq!(shape_to_vec(&[2, -1, 4], 24).unwrap(), vec![2, 3, 4]);
        assert_eq!(shape_to_vec(&[2, 3, -1], 24).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn test_shape_to_vec_err() {
        // invalid extent size
        assert_eq!(
            shape_to_vec(&[1, 3, -4], 24).expect_err(""),
            ShapeError::InvalidExtent(-4)
        );
        assert_eq!(
            shape_to_vec(&[1, 0, 4], 24).expect_err(""),
            ShapeError::InvalidExtent(0)
        );

        // size mismatch
        assert_eq!(
            shape_to_vec(&[1, 3, 4], 24).expect_err(""),
            ShapeError::SizeMismatch(24, 12)
        );
        assert_eq!(
            shape_to_vec(&[1, 3, 4], 23).expect_err(""),
            ShapeError::SizeMismatch(23, 12)
        );
        assert_eq!(
            shape_to_vec(&[1, 3, 4], 25).expect_err(""),
            ShapeError::SizeMismatch(25, 12)
        );

        // infer two times
        assert_eq!(
            shape_to_vec(&[-1, -1, 4], 24).expect_err(""),
            ShapeError::InvalidInference
        );

        // use infer but no specified size
        assert_eq!(
            shape_to_vec(&[-1, 3, 4], 0).expect_err(""),
            ShapeError::InvalidInference
        );

        // not divisible infer size
        assert_eq!(
            shape_to_vec(&[-1, 3, 4], 25).expect_err(""),
            ShapeError::InvalidInference
        );
    }

    #[test]
    fn test_union() {
        assert_eq!(union([1, 3, 4], [1, 3, 4]).unwrap(), vec![1, 3, 4]);
        assert_eq!(
            union([5, 6, 1, 3, 4], [1, 3, 4]).unwrap(),
            vec![5, 6, 1, 3, 4]
        );
        assert_eq!(
            union([5, 6, 1, 3, 4], [7, 3, 4]).unwrap(),
            vec![5, 6, 7, 3, 4]
        );
        assert_eq!(
            union([1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1]).unwrap(),
            vec![2, 2, 2, 2, 2, 2]
        );
    }
}
