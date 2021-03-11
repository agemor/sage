// simple tensor operations... from scratch!

use std::cell::UnsafeCell;
use std::fmt::Formatter;
use std::rc::Rc;
use std::{fmt, ops, ptr, slice};

use itertools::{zip, Itertools};
use matrixmultiply;
use ndarray::ShapeBuilder;
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::Normal;

use crate::tensor::iter::{AlongAxisIter, Iter, IterMut};
use crate::tensor::shape::{Shape, ShapeError, ToIndex, ToShape};

pub mod init;
pub mod iter;
mod ops;
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
    arr: Rc<UnsafeCell<Vec<f32>>>,
}

impl Tensor {
    /////////////////////// basic functions ///////////////////////

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    // number of dimensions (also known as .. order, degree, ndims)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    // total number of elements in this tensor
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    pub fn arr(&self) -> &Vec<f32> {
        unsafe { self.arr.get().as_ref() }.unwrap()
    }
    pub fn arr_mut(&self) -> &mut Vec<f32> {
        unsafe { self.arr.get().as_mut() }.unwrap()
    }

    pub fn mem_size(&self) -> usize {
        self.arr().len()
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

    // constructors
    pub fn from_slice<S>(shape: S, slice: &[f32]) -> Self
    where
        S: ToShape,
    {
        Tensor::from_vec(shape, slice.to_vec())
    }

    pub fn from_vec<S>(shape: S, v: Vec<f32>) -> Self
    where
        S: ToShape,
    {
        // materialization of shape
        let shape = shape.to_shape(v.len());
        let strides = Shape::default_strides(shape);

        Tensor {
            shape,
            strides,
            offset: 0,
            arr: Rc::new(UnsafeCell::new(v)),
        }
    }

    // Create tensor from single element
    pub fn from_elem<S>(shape: S, elem: f32) -> Self
    where
        S: ToShape,
    {
        let shape = shape.to_shape(0);
        let v = vec![elem; shape.size()];
        Tensor::from_vec(shape, v)
    }

    // Create tensor from the given distribution
    pub fn from_dist<S, T>(shape: S, dist: T) -> Tensor
    where
        S: ToShape,
        T: Distribution<f32>,
    {
        let shape = shape.to_shape(0);

        let mut v = Vec::<f32>::with_capacity(shape.size());
        let mut rng = thread_rng(); //&mut SmallRng::from_rng(thread_rng()).unwrap();

        for _ in 0..shape.size() {
            v.push(dist.sample(&mut rng));
        }

        Tensor::from_vec(shape, v)
    }

    // private
    fn from_ndarray(ndarray: ndarray::ArrayD<f32>) -> Self {
        let shape = ndarray.shape().to_shape(0);
        let strides = ndarray
            .strides()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<usize>>();

        let vec = ndarray.into_raw_vec();

        // gather ndarray strides and apply those strides to resulting tensor
        let mut t = Tensor::from_vec(shape, vec);
        t.strides = strides;

        t
    }

    pub fn scalar(v: f32) -> Self {
        Tensor::from_elem([1], v)
    }

    // only for the debugging
    pub fn null() -> Self {
        Tensor::scalar(0.0)
    }

    fn view<S>(orig: &Tensor, shape: S, strides: &[usize], offset: usize) -> Tensor
    where
        S: ToShape,
    {
        Tensor {
            shape: shape.to_shape(orig.size()),
            strides: strides.to_vec(),
            offset,
            arr: orig.arr.clone(), // this increases reference counter (do not copy actual data)
        }
    }

    ////////////////// ndarray view converter (private) ////////////////

    fn to_ndarray(&self) -> ndarray::ArrayViewD<f32> {
        unsafe {
            // let inner = self.arr.get().as_ref().unwrap();

            let mut raw_ptr = self.arr().as_ptr();
            // calculate offset
            raw_ptr = raw_ptr.add(self.offset);

            let shape = ndarray::IxDyn(self.shape.sizes()).strides(ndarray::IxDyn(self.strides()));

            ndarray::ArrayViewD::from_shape_ptr(shape, raw_ptr)
        }
    }

    ////////////////////// optimizers //////////////////////

    fn shrink_if_possible(&self) {
        if Rc::strong_count(&self.arr) == 1 {
            // for cases where actual_size < current_size (i.e., broadcast), we does nothing special.
            if self.mem_size() > self.size() {
                self.standalone();
            }
        }
    }

    pub fn standalone(&self) {
        let owned = self.to_ndarray().to_owned();
        let new_v = owned.into_raw_vec();
        unsafe {
            *self.arr.get() = new_v;
        }
    }

    ///////////////// init constructors /////////////////

    pub fn zeros<S>(shape: S) -> Tensor
    where
        S: ToShape,
    {
        Tensor::from_elem(shape, 0.0)
    }

    pub fn ones<S>(shape: S) -> Tensor
    where
        S: ToShape,
    {
        Tensor::from_elem(shape, 1.0)
    }

    // sample from standard normal dist
    pub fn randn<S>(shape: S) -> Tensor
    where
        S: ToShape,
    {
        Tensor::from_dist(shape, Normal::new(0.0, 1.0).unwrap())
    }

    ///////////////// comparison /////////////////
    pub fn equals(&self, tensor: &Tensor, eps: f32) -> bool {
        let eq_map = self
            .zip_map(
                tensor,
                |&a, &b| if (a - b).abs() <= eps { 0 } else { 1 } as f32,
            )
            .unwrap();

        eq_map.is_zero()
    }

    pub fn is_zero(&self) -> bool {
        self.logical_iter().all(|&x| x == 0.0)
    }

    ///////////////// index, slice, join /////////////////

    ////////// Iterator////////////

    pub fn along_axis<I>(&self, axis: I) -> AlongAxisIter
    where
        I: ToIndex,
    {
        let axis = axis.to_index(self.rank());
        AlongAxisIter::new(self, axis)
    }

    pub fn fold_axis<I, F>(&self, axis: I, init: f32, mut fold: F) -> Tensor
    where
        I: ToIndex,
        F: FnMut(&f32, &f32) -> f32,
    {
        let axis_u = axis.to_index(self.rank());

        let mut folded_shape = self.shape;
        folded_shape.remove(axis_u);
        let folded = Tensor::from_elem(folded_shape, init);

        for t in self.along_axis(axis) {
            // wow!
            folded
                .random_iter_mut()
                .zip(t.logical_iter())
                .for_each(|(x, y)| {
                    *x = fold(x, y);
                })
        }
        folded
    }

    ////////////// modifier /////////////////

    // preserve original memory layout??
    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(&f32) -> f32,
    {
        let v =
            // fast, contiguous iter
            if self.is_bijective() && self.is_contiguous() {
                unsafe {
                    let offset_ptr = self.arr().as_ptr().add(self.offset);
                    let s = slice::from_raw_parts(offset_ptr, self.size());
                    s.iter().map(f).collect::<Vec<f32>>()
                }
            }
            // non-bijective tensors have to use random_iter (with current strides)
            // or logical_iter (with default strides)
            else {
                self.logical_iter().map(f).collect::<Vec<f32>>()
                // ^--- allows new tensor initialized with default strides
            };

        Tensor::from_vec(self.shape(), v)
    }

    pub fn mapv_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(&f32) -> f32,
    {
        if self.is_contiguous() {
            unsafe {
                let offset_ptr = self.arr_mut().as_mut_ptr().add(self.offset);
                let s = slice::from_raw_parts_mut(offset_ptr, self.size());
                s.iter_mut().for_each(|x| *x = f(x));
            }
        }
        // order does not matter..  every element is only visited once.
        else {
            self.random_iter_mut().for_each(|x| *x = f(x));
        }
    }

    // visits each elements in a logical order.
    // used for pairwise tensor operations.
    pub fn logical_iter(&self) -> Iter {
        Iter::new(
            self.arr().as_ptr(),
            self.offset,
            self.shape.sizes(),
            self.strides(),
        )
    }

    // Visits inner elements in an arbitrary order, but faster than logical iterator
    // because it visits elements in a way that cpu cache is utilized.
    pub fn random_iter_mut(&self) -> IterMut {
        let (strides, dim): (Vec<usize>, Vec<usize>) = self
            .strides
            .iter()
            .zip(self.shape.iter())
            .filter(|(&s, _)| s > 0)
            .sorted_by(|(&s1, _), (&s2, _)| s2.cmp(&s1))
            .unzip();

        IterMut::new(self.arr_mut().as_mut_ptr(), self.offset, &dim, &strides)
    }

    pub fn zip_map<F>(&self, other: &Tensor, f: F) -> Result<Tensor, ShapeError>
    where
        F: Fn(&f32, &f32) -> f32,
    {
        if let Ok(u) = Shape::union(self.shape, other.shape) {
            let a = self.upcast(&u).unwrap();
            let b = other.upcast(&u).unwrap();

            let v = a
                .logical_iter()
                .zip(b.logical_iter())
                .map(|(a, b)| f(a, b))
                .collect::<Vec<f32>>();

            Ok(Tensor::from_vec(u, v))
        } else {
            Err(ShapeError::new("cannot broadcast!"))
        }
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
        self.logical_iter()
            .zip(other.logical_iter())
            .all(|(a, b)| a == b)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let v = self.logical_iter().copied().collect::<Vec<f32>>();
        Tensor::from_vec(self.shape(), v)
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

// let user know it is dangerous and might cause UD.

// The reason we are not taking moved value is that,
// i) cannot ensure it's big enough (or too big) to handle new value (due to broadcasting)
// ii) it will be eventually dropped at the end of the scope (if reference count becomes 0)

macro_rules! impl_tensor_op {
    ($op:tt, $f:expr) => {
        impl_op!($op |a: Tensor, b: Tensor| -> Tensor {$f(&a, &b) });
        impl_op!($op |a: &Tensor, b: Tensor| -> Tensor {  $f(a, &b) });
        impl_op!($op |a: Tensor, b: &Tensor| -> Tensor {  $f(&a, b) });
        impl_op!($op |a: &Tensor, b: &Tensor| -> Tensor {  $f(a, b) });
        impl_op!($op |a: Tensor, b: f32| -> Tensor {  $f(&a, &Tensor::scalar(b)) });
        impl_op!($op |a: &Tensor, b: f32| -> Tensor {  $f(a, &Tensor::scalar(b))});
        impl_op!($op |a: f32, b: Tensor| -> Tensor {  $f(&Tensor::scalar(a), &b) });
        impl_op!($op |a: f32, b: &Tensor| -> Tensor {  $f(&Tensor::scalar(a), b)});
    }
}

// basic arithmetics
impl_tensor_op!(+, math::add);
impl_tensor_op!(-, math::sub);
impl_tensor_op!(*, math::mul);
impl_tensor_op!(/, math::div);

impl_op!(-|a: Tensor| -> Tensor { math::neg(&a) });
impl_op!(-|a: &Tensor| -> Tensor { math::neg(a) });

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
        assert_eq!(Tensor::zeros([]).rank(), 0);
        assert_eq!(Tensor::zeros([1, 1, 4, 10]).rank(), 4);
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
            Tensor::cat(&[&a, &b], 2).unwrap().shape(),
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
        assert_eq!(a.squeeze(-1).shape(), [1, 3, 2].to_shape(0));
        assert_eq!(a.squeeze(0).shape(), [3, 2, 1].to_shape(0));
    }

    #[test]
    #[should_panic]
    fn test_squeeze_panic() {
        let a = Tensor::ones([1, 3, 2, 1]);
        a.squeeze(2);
    }

    #[test]
    fn test_expand_dims() {
        let a = Tensor::ones([3, 2, 9]);
        assert_eq!(a.expand_dims(3).shape(), [3, 2, 9, 1].to_shape(0));
        assert_eq!(a.expand_dims(-1).shape(), [3, 2, 9, 1].to_shape(0));
        assert_eq!(a.expand_dims(0).shape(), [1, 3, 2, 9].to_shape(0));
        assert_eq!(a.expand_dims(1).shape(), [3, 1, 2, 9].to_shape(0));
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
