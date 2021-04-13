// simple tensor operations... from scratch!

use std::cell::UnsafeCell;
use std::fmt::Formatter;
use std::rc::Rc;
use std::{fmt, ptr, slice};

use itertools::{zip, Itertools};
use ndarray::ShapeBuilder;
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::Normal;

use crate::tensor::backend::{Backend, BinaryOperation, Buffer};
use crate::tensor::iter::{AlongAxisIter, Iter, IterMut};
use crate::tensor::shape::{Shape, ShapeError, ToIndex, ToIndices, ToShape};

pub mod backend;
pub mod init;
pub mod iter;
pub mod ops;
pub mod shape;

/////////////////////////// Tensor implementation ///////////////////////////

pub struct Tensor {
    // keep it private

    // tensor dim
    shape: Shape,

    // array offset and strides
    strides: Vec<usize>,
    offset: usize,

    // underlying array that holds actual data
    buffer: Rc<Buffer>,
}

impl Tensor {
    /////////////////////// basic functions ///////////////////////

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn extents(&self) -> &[usize] {
        self.shape.sizes()
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    // number of dimensions (also known as .. order, degree, ndims)
    pub fn order(&self) -> usize {
        self.shape.len()
    }

    // total number of elements in this tensor
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    pub fn backend(&self) -> &dyn Backend {
        self.buffer.backend()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.buffer.to_vec()
    }

    pub fn equals(&self, other: &Tensor) -> bool {
        let eq = self.binary_op(other, BinaryOperation::NotEq);
        eq.sum((0..self.order()).collect_vec(), false).to_vec()[0] == 0.0
    }

    // returns that the inner content of the tensor is contiguous.
    // contiguous in memory space
    // This method asks: Did user do any indexing (slicing) operations?
    pub fn is_contiguous(&self) -> bool {
        // independent to the broadcast and reshaping operations.

        // remove 0 from stride (remove the effect from the broadcast)
        let mut bij_shape = Vec::new(); // 'bijective (no-broadcast)' dimension
        let mut bij_strides = Vec::new();

        for (s, d) in zip(&self.strides, self.shape) {
            // count only 'real' dims
            if *s > 0 {
                bij_shape.push(d);
                bij_strides.push(*s);
            }
        }
        // sort descending order (remove effects of reshaping operations)
        bij_strides.sort_by(|a, b| b.cmp(a));

        Shape::default_strides(bij_shape) == bij_strides
    }

    // one-to-one correspondence
    // This method asks: Did user do any broadcasting operations?
    pub fn is_bijective(&self) -> bool {
        self.strides.iter().all(|&a| a > 0)
    }

    // This method asks: did the user used any reshape operations?
    pub fn is_ordered(&self) -> bool {
        self.strides
            .iter()
            .filter(|&a| *a > 0)
            .is_sorted_by(|&a, &b| Some(b.cmp(a)))
    }

    ///////////////// constructors /////////////////

    fn uninit<S>(shape: S, backend: &dyn Backend) -> Self
    where
        S: ToShape,
    {
        let shape = shape.to_shape(0);
        let strides = Shape::default_strides(shape);

        Tensor {
            shape,
            strides,
            offset: 0,
            buffer: Rc::new(backend.alloc_mem(shape.size())),
        }
    }

    pub fn from_iter<S, I>(shape: S, data: I, backend: &dyn Backend) -> Self
    where
        S: ToShape,
        I: ExactSizeIterator<Item = f32>,
    {
        let shape = shape.to_shape(data.len());
        let strides = Shape::default_strides(shape);

        Tensor {
            shape,
            strides,
            offset: 0,
            buffer: Rc::new(backend.alloc_mem_from_iter(data)),
        }
    }

    // Create tensor from single element
    pub fn from_elem<S>(shape: S, elem: f32, backend: &dyn Backend) -> Self
    where
        S: ToShape,
    {
        let shape = shape.to_shape(0);
        let iter = (0..shape.size()).map(|_| elem);

        Tensor::from_iter(shape, iter, backend)
    }

    // Create tensor from the given distribution
    pub fn from_dist<S, D>(shape: S, dist: D, backend: &dyn Backend) -> Tensor
    where
        S: ToShape,
        D: Distribution<f32>,
    {
        let shape = shape.to_shape(0);

        let mut rng = thread_rng();
        let iter = (0..shape.size()).map(|_| dist.sample(&mut rng));

        Tensor::from_iter(shape, iter, backend)
    }

    pub fn scalar(v: f32, backend: &dyn Backend) -> Self {
        Tensor::from_elem([1], v, backend)
    }

    fn view<S>(source: &Tensor, shape: S, strides: &[usize], offset: usize) -> Tensor
    where
        S: ToShape,
    {
        Tensor {
            shape: shape.to_shape(0), // no check
            strides: strides.to_vec(),
            offset,
            buffer: source.buffer.clone(), // this increases reference counter (do not copy actual data)
        }
    }

    ///////////////// init constructors /////////////////

    pub fn zeros<S>(shape: S, backend: &dyn Backend) -> Tensor
    where
        S: ToShape,
    {
        Tensor::from_elem(shape, 0.0, backend)
    }

    pub fn ones<S>(shape: S, backend: &dyn Backend) -> Tensor
    where
        S: ToShape,
    {
        Tensor::from_elem(shape, 1.0, backend)
    }

    // sample from standard normal dist
    pub fn randn<S>(shape: S, backend: &dyn Backend) -> Tensor
    where
        S: ToShape,
    {
        Tensor::from_dist(shape, Normal::new(0.0, 1.0).unwrap(), backend)
    }

    ////////////////////// optimizers //////////////////////

    // fn shrink_if_possible(&self) {
    //     if Rc::strong_count(&self.arr) == 1 {
    //         // for cases where actual_size < current_size (i.e., broadcast), we does nothing special.
    //         if self.mem_size() > self.size() {
    //             self.standalone();
    //         }
    //     }
    // }
    //
    // pub fn standalone(&self) {
    //     let owned = self.to_ndarray().to_owned();
    //     let new_v = owned.into_raw_vec();
    //     unsafe {
    //         *self.arr.get() = new_v;
    //     }
    // }

    //////////////////////////////////////////////////////////////

    ///////////////// comparison /////////////////

    ///////////////// index, slice, join /////////////////

    ////////// Iterator////////////

    pub fn along_axis<I>(&self, axis: I) -> AlongAxisIter
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.order());
        AlongAxisIter::new(self, axis)
    }
}

impl Eq for Tensor {}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        // check shape
        if self.shape() != other.shape() {
            return false;
        }

        // check inner contents
        self.equals(other)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        self.recreate()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        //writeln!(f, "{:?}", unsafe { self.inner.get().as_ref() }.unwrap());
        writeln!(f, "{}", self.to_ndarray())
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("dim", &self.shape)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("arr\n", &self.to_ndarray())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq() {
        assert_eq!(
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 12.,]),
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 12.,])
        );

        assert_ne!(
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 12.,]),
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 13.,])
        );

        assert_ne!(
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 12.,]),
            Tensor::from_slice([1, 1, 1, 3, 2], &[1., 4., 6., 8., 10., 12.,])
        );
    }

    #[test]
    fn test_op() {
        let a = Tensor::from_slice(
            [2, 3, 2],
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        let b = Tensor::from_slice([3, 2], &[1., 2., 3., 4., 5., 6.]);

        let c = &a + &b;

        assert_ne!(
            c,
            Tensor::from_slice(
                [2, 3, 2],
                &[1., 4., 6., 8., 10., 12., 8., 10., 12., 14., 16., 19.],
            )
        );

        assert_eq!(
            c,
            Tensor::from_slice(
                [2, 3, 2],
                &[2., 4., 6., 8., 10., 12., 8., 10., 12., 14., 16., 18.],
            )
        );

        let c = -&b;

        assert_ne!(
            c,
            Tensor::from_slice([3, 2], &[-1., -2., -3., -4., -5., 6.])
        );

        assert_eq!(
            c,
            Tensor::from_slice([3, 2], &[-1., -2., -3., -4., -5., -6.])
        );

        assert_eq!(Tensor::scalar(3.0) + 4.0, Tensor::scalar(7.0));
    }

    #[test]
    fn test_shape() {
        let a = Tensor::zeros([1, 4, 10]);
        assert_eq!(a.shape(), [1, 4, 10].to_shape(0));
        assert_eq!(a.shape[0], 1);
        assert_eq!(a.shape[1], 4);
        assert_eq!(a.shape[2], 10);
    }

    #[test]
    fn test_strides() {}

    #[test]
    fn test_rank() {
        assert_eq!(Tensor::zeros([]).order(), 0);
        assert_eq!(Tensor::zeros([1, 1, 4, 10]).order(), 4);
    }

    #[test]
    fn test_is_contiguous() {}

    #[test]
    fn test_from_slice() {}

    #[test]
    fn test_from_vec() {}

    #[test]
    fn test_from_elem() {}

    #[test]
    fn test_from_dist() {}

    #[test]
    fn test_view() {}

    #[test]
    fn test_from_ndarray() {}

    #[test]
    fn test_ndarray() {
        let a = Tensor::randn([2, 3, 4]);
        println!("{:?}", a.to_ndarray());

        let b = a.transpose(-1, -2);
        println!("{:?}", b.to_ndarray());
    }

    #[test]
    fn test_shrink_if_possible() {}

    #[test]
    fn test_standalone() {}

    #[test]
    fn test_zeros() {}

    #[test]
    fn test_randn() {}

    #[test]
    fn test_cat() {
        let a = Tensor::ones([3, 2, 5]);
        let b = Tensor::ones([3, 2, 5]);
        assert_eq!(
            Tensor::concat(&[&a, &b], 2).unwrap().shape(),
            [3, 2, 10].to_shape(0)
        );
    }

    #[test]
    fn test_stack() {
        let a = Tensor::ones([3, 2, 5]);
        let b = Tensor::ones([3, 2, 5]);
        assert_eq!(
            Tensor::stack(&[&a, &b], 2).unwrap().shape(),
            [3, 2, 2, 5].to_shape(0)
        );
    }

    #[test]
    fn test_squeeze() {
        let a = Tensor::ones([1, 3, 2, 1]);
        assert_eq!(a.squeeze_axis(-1).shape(), [1, 3, 2].to_shape(0));
        assert_eq!(a.squeeze_axis(0).shape(), [3, 2, 1].to_shape(0));
    }

    #[test]
    #[should_panic]
    fn test_squeeze_panic() {
        let a = Tensor::ones([1, 3, 2, 1]);
        a.squeeze_axis(2);
    }

    #[test]
    fn test_expand_dims() {
        let a = Tensor::ones([3, 2, 9]);
        assert_eq!(a.expand_axis(3).shape(), [3, 2, 9, 1].to_shape(0));
        assert_eq!(a.expand_axis(-1).shape(), [3, 2, 9, 1].to_shape(0));
        assert_eq!(a.expand_axis(0).shape(), [1, 3, 2, 9].to_shape(0));
        assert_eq!(a.expand_axis(1).shape(), [3, 1, 2, 9].to_shape(0));
    }

    #[test]
    fn test_reshape() {}

    #[test]
    fn test_transpose() {}

    #[test]
    fn test_permute() {}

    #[test]
    fn test_upcast() {
        let a = Tensor::ones([3, 1, 9]);
        assert_eq!(a.upcast([3, 1, 9]).unwrap().shape(), [3, 1, 9].to_shape(0));
        assert_eq!(a.upcast([3, 1, 9]).unwrap(), Tensor::ones([3, 1, 9]));

        assert_eq!(
            a.upcast([10, 3, 7, 9]).unwrap().shape(),
            [10, 3, 7, 9].to_shape(0)
        );
        assert_eq!(
            a.upcast([10, 3, 7, 9]).unwrap(),
            Tensor::ones([10, 3, 7, 9])
        );
    }

    #[test]
    #[should_panic]
    fn test_upcast_panic() {
        let a = Tensor::ones([3, 1, 9]);
        a.upcast([3, 1, 8]).unwrap();
        a.upcast([5, 5, 9]).unwrap();
    }

    #[test]
    fn test_index_axis() {}

    #[test]
    fn test_index() {}

    #[test]
    fn test_slice_axis() {}

    #[test]
    fn test_slice() {}

    #[test]
    fn test_along_axis() {
        let a = Tensor::ones([3, 1, 9]);
        a.along_axis(-1)
            .for_each(|t| assert_eq!(t.shape(), [3, 1].to_shape(0)));
        a.along_axis(0)
            .for_each(|t| assert_eq!(t.shape(), [1, 9].to_shape(0)));
    }

    #[test]
    fn test_fold_axis() {
        let a = Tensor::ones([3, 1, 9]);

        let b = a.fold_axis(-1, 0.0, |&a, &b| a + b);

        assert_eq!(b, Tensor::from_elem([3, 1], 9.0));
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_slice(
            [3, 2, 3],
            &[
                -1.33820, -0.27636, -1.79478, -0.50638, -1.83333, -0.31560, 1.58321, 1.34656,
                -1.13917, 1.21047, -1.21295, -1.94985, 0.32884, -0.65730, -0.34635, 1.41366,
                -0.82830, -1.94889,
            ],
        );

        let b = Tensor::from_slice([3, 1], &[0.89976, -1.99751, 0.03085]);

        let c = Tensor::from_slice(
            [3, 2, 1],
            &[-0.70741, 3.19675, -1.30040, 3.45186, 1.59816, 2.86636],
        );

        assert!(a.matmul(&b).equals(&c, 0.001));
    }

    #[test]
    fn test_matvec() {
        let a = Tensor::from_slice([2, 2, 2], &[1., 2., 3., 4., 3., 4., 5., 6.]);

        let b = Tensor::from_slice([2, 2], &[1., 2., 3., 1.]);

        assert_eq!(
            a.matvec(&b),
            Tensor::from_slice([2, 2], &[5., 11., 13., 21.,])
        );
    }

    #[test]
    fn test_mapv() {}

    #[test]
    fn test_to_index() {
        assert_eq!((-3).to_index(3), 0);
        assert_eq!((-2).to_index(3), 1);
        assert_eq!((-1).to_index(3), 2);
        assert_eq!((0).to_index(3), 0);
        assert_eq!((1).to_index(3), 1);
        assert_eq!((2).to_index(3), 2);
    }

    #[test]
    #[should_panic]
    fn test_to_index_panic() {
        assert_eq!((-4).to_index(3), 0);
        assert_eq!((3).to_index(3), 2);
    }
}
