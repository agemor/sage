// simple tensor operations... from scratch!

use std::cell::UnsafeCell;
use std::fmt::Formatter;
use std::rc::Rc;
use std::{fmt, ops, slice};

use itertools::{zip, Itertools};
use ndarray::ShapeBuilder;
use rand::distributions::Distribution;
use rand::rngs::SmallRng;
use rand::{thread_rng, SeedableRng};
use rand_distr::Normal;

use crate::tensor::iter::{AlongAxisIter, Iter, IterMut};
use crate::shape::{Shape, ToShape, ShapeError, ToIndex};

pub mod init;
pub mod iter;
pub mod math;

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
        unsafe { self.arr().len() }
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
            .is_sorted_by(|&a, &b| Some({ b.cmp(a) }))
    }

    // constructors
    pub fn from_slice<S>(shape: S, slice: &[f32]) -> Self
        where S: ToShape
    {
        Tensor::from_vec(shape, slice.to_vec())
    }

    pub fn from_vec<S>(shape: S, v: Vec<f32>) -> Self
        where S: ToShape,
    {
        // materialization of shape
        let shape = shape.to_shape();
        let strides = Shape::default_strides(shape);

        // check shape-vector compatibility
        if shape.size() != v.len() {
            panic!("shape and vector not compatible")
        }

        Tensor {
            shape,
            strides,
            offset: 0,
            arr: Rc::new(UnsafeCell::new(v)),
        }
    }


    // Create tensor from single element
    pub fn from_elem<S>(shape: S, elem: f32) -> Self
        where S: ToShape,
    {
        let shape = shape.to_shape();
        let v = vec![elem; shape.size()];
        Tensor::from_vec(shape, v)
    }

    // Create tensor from the given distribution
    pub fn from_dist<S, T>(shape: S, dist: T) -> Tensor
        where S: ToShape,
              T: Distribution<f32>,
    {
        let shape = shape.to_shape();

        let mut v = Vec::<f32>::with_capacity(shape.size());
        let mut rng = &mut SmallRng::from_rng(thread_rng()).unwrap();

        for _ in 0..shape.size() {
            v.push(dist.sample(rng));
        }

        Tensor::from_vec(shape, v)
    }

    // private
    fn from_ndarray(ndarray: ndarray::ArrayD<f32>) -> Self {
        let vec = ndarray.into_raw_vec();
        Tensor::from_vec(ndarray.shape(), vec) // must unwrap
    }

    pub fn scalar(v: f32) -> Self {
        Tensor::from_elem([1], v)
    }


    fn view<S>(orig: &Tensor, shape: S, strides: &[usize], offset: usize) -> Tensor
        where S: ToShape,
    {
        Tensor {
            shape: shape.to_shape(),
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
            raw_ptr = raw_ptr.offset(self.offset as isize);

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
        where S: ToShape
    {
        Tensor::from_elem(shape, 0.0)
    }

    pub fn ones<S>(shape: S) -> Tensor
        where S: ToShape
    {
        Tensor::from_elem(shape, 1.0)
    }

    // sample from standard normal dist
    pub fn randn<S>(shape: S) -> Tensor
        where S: ToShape
    {
        Tensor::from_dist(shape, Normal::new(0.0, 1.0).unwrap())
    }

    ///////////////// index, slice, join /////////////////
    // * Creates new tensor
    pub fn cat<I>(tensors: &[Tensor], axis: I) -> Result<Tensor, ShapeError>
        where I: ToIndex
    {
        let axis = axis.to_index(tensors[0].rank());

        // TODO: spit out some errors when tensors in different shape, except in the cat axis.

        // convert into array views
        let t_data = tensors
            .iter()
            .map(|t| t.to_ndarray())
            .collect::<Vec<ndarray::ArrayViewD<f32>>>();

        if let Ok(arr) = ndarray::concatenate(ndarray::Axis(axis), &t_data) {
            Ok(Tensor::from_ndarray(arr))
        } else {
            Err(ShapeError::new("cannot concat!"))
        }
    }

    // * Creates new tensor
    pub fn stack<I>(tensors: &[Tensor], axis: I) -> Result<Tensor, ShapeError>
        where I: ToIndex
    {
        if !tensors.iter().map(|e| e.shape).all_equal() {
            panic!("all tensors should be in the same shape");
        }

        let axis = axis.to_index(tensors[0].rank() + 1);

        // convert into array views
        let t_data = tensors
            .iter()
            .map(|t| t.to_ndarray())
            .collect::<Vec<ndarray::ArrayViewD<f32>>>();
        if let Ok(arr) = ndarray::stack(ndarray::Axis(axis), &t_data) {
            Ok(Tensor::from_ndarray(arr))
        } else {
            Err(ShapeError::new("cannot stack!"))
        }
    }

    pub fn squeeze<I>(&self, axis: I) -> Tensor
        where I: ToIndex
    {
        let axis = axis.to_index(self.rank());

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        if new_shape[axis] == 1 {
            new_shape.remove(axis);
            new_strides.remove(axis);
        } else {
            panic!("dim=1 cannot be squeezed");
        }
        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    pub fn expand_dims<I>(&self, axis: I) -> Tensor
        where I: ToIndex
    {
        // allow unexisting index
        let axis = axis.to_index(self.rank() + 1);

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        new_shape.insert(axis, 1);

        if new_strides.len() == axis {
            new_strides.push(1);
        } else {
            let s = self.strides[axis];
            new_strides.insert(axis, s);
        }

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    // reshape (underlying data does not change)
    pub fn reshape<S>(&self, shape: S) -> Result<Tensor, ShapeError>
        where S: ToShape
    {
        let new_shape = shape.to_shape();
        let new_strides = Shape::default_strides(new_shape);

        if self.is_contiguous() {
            if self.size() == new_shape.size() {
                Ok(Tensor::view(self, new_shape, &new_strides, self.offset))
            } else {
                Err(ShapeError::new("shape not compatible"))
            }
        } else {
            Err(ShapeError::new("tensor not contiguous"))
        }
    }

    // swap last two dims of tensor
    pub fn transpose<I, J>(&self, axis_a: I, axis_b: J) -> Tensor
        where I: ToIndex,
              J: ToIndex
    {
        let axis_a = axis_a.to_index(self.rank());
        let axis_b = axis_b.to_index(self.rank());

        if axis_a == axis_b {
            panic!("same axis");
        }

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        new_shape.swap(axis_a, axis_b);
        new_strides.swap(axis_a, axis_b);

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    pub fn permute<I>(&self, axes: &[I]) -> Tensor
        where I: ToIndex
    {
        let axes = axes
            .iter()
            .map(|axis| axis.to_index(self.rank()))
            .collect::<Vec<usize>>();

        let mut use_counts = vec![0; self.rank()];

        axes.iter().for_each(|axis| {
            use_counts[*axis] += 1;
        });

        if use_counts.iter().any(|count| *count != 1) {
            panic!("some axes are not used, or used more than once");
        }

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        for axis in axes {
            new_shape[axis] = self.shape[axis];
            new_strides[axis] = self.strides[axis];
        }

        Tensor::view(self, new_shape, &new_strides, self.offset)
    }

    pub fn upcast<S>(&self, shape: S) -> Result<Tensor, ShapeError>
        where
            S: ToShape,
    {
        let target_shape = shape.to_shape();

        if self.rank() > target_shape.len() {
            return Err(ShapeError::new("invalid broadcast.. too small shape"));
        }

        // a = ( 10, 7)
        // 0 ... 69

        // a[3][2] = 3 * 7 + 2 * 1

        // a = (10, 1, 7)
        // 0 ... 69
        // a[3][0][2] = 3 * 7 + 0 * 1 + 2 * 1

        // a = (1, 10, 7)
        // 0 ... 69
        // a[0][3][2] = 0 * 70 + 3 * 7

        // in broadcasting we set 0 for augmented dimension.

        // create some fake strides

        // (3, 1, 5) broadcast (2, 1, 1, 9, 5)

        // padded:      (1, 1, 3, 1, 5)

        // union shape: (2, 1, 3, 9, 5)  <----- this is given as a parameter

        // stride       (0, 0, k, 0, k)
        //                        ^________ always 0 anyways...

        // pad 1 for dim

        let mut new_shape = self.shape;
        let mut new_strides = self.strides.clone();

        let pad_len = target_shape.len() - new_shape.len();

        for _ in 0..pad_len {
            new_shape.insert(0, 1);
            new_strides.insert(0, 0);
        }

        for ((a, b), s) in new_shape
            .iter_mut()
            .zip(target_shape.iter())
            .zip(new_strides.iter_mut())
        {
            if *a != *b {
                // for broadcast axes, 'mute' them by set its stride to zero
                if *a == 1 {
                    *a = *b;
                    *s = 0;
                } else {
                    return Err(ShapeError::new(
                        "invalid broadcast... target shape should be larger.",
                    ));
                }
            }
        }
        Ok(Tensor::view(self, new_shape, &new_strides, self.offset))
    }

    /// make possible a[..][3][..] kind of operations.
    pub fn index_axis<I, J>(&self, index: I, axis: J) -> Tensor
        where I: ToIndex,
              J: ToIndex
    {
        let axis = axis.to_index(self.rank());
        let index = index.to_index(self.shape[axis]);


        // a = (10, 10, 10)
        // a[2]
        // stride = (100, 10, 1)
        // offset = 2 * 100

        // a[2][3]
        // offset = 2 * 100 + 3 * 10

        // b = a[:][3]
        // b[7][5]
        // a offset = 3 * 10
        // b... 7 * 100 + offset + 5 * 1     ----> just remove axis

        let offset = self.strides[axis] * index;

        let mut new_shape = self.shape;
        let mut new_stride = self.strides.clone();

        new_shape.remove(axis);
        new_stride.remove(axis);

        Tensor::view(self, new_shape, &new_stride, self.offset + offset)
    }

    // single index.. unwraps first one.
    pub fn index<I>(&self, index: I) -> Tensor
        where I: ToIndex
    {
        self.index_axis(index, 0)
    }

    // make possible of a[..][2..3][..] kind of operations
    pub fn slice_axis<I, J, K>(&self, start_index: I, end_index: J, axis: K) -> Tensor
        where I: ToIndex,
              J: ToIndex,
              K: ToIndex
    {
        let axis = axis.to_index(self.rank());

        let start_index = start_index.to_index(self.shape[axis]);
        let end_index = end_index.to_index(self.shape[axis]);

        if start_index > end_index {
            panic!("start and end index are not in the order");
        }


        // a = (10, 10, 10)
        // a[2]
        // stride = (100, 10, 1)
        // offset = 2 * 100

        // a[2][3]
        // offset = 2 * 100 + 3 * 10

        // b = a[:][3]
        // b[7][5]
        // a offset = 3 * 10
        // b... 7 * 100 + offset + 5 * 1     ----> just remove axis
        let stride = self.strides[axis];
        let offset = stride * start_index;

        let mut new_dim = self.shape.clone();
        let mut new_stride = self.strides.clone();

        new_dim[axis] = end_index - start_index + 1;
        new_stride.remove(axis);

        Tensor::view(self, new_dim, &new_stride, self.offset + offset)
    }

    pub fn slice<I, J>(&self, start_index: I, end_index: J) -> Tensor
        where I: ToIndex,
              J: ToIndex
    {
        self.slice_axis(start_index, end_index, 0)
    }

    // * Creates new tensor
    // (broadcast) batch matrix multiplication
    // e.g., (3, 5, 1, 7, 11) matmul (9, 11, 2) = > (3, 5, 9, 7, 2)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let rank_a = self.rank();
        let rank_b = other.rank();

        // ensure at least two dims
        if rank_a < 2 || rank_b < 2 {
            panic!("not a matrix");
        }

        // check last two dims are compatible,
        if self.shape[rank_a - 1] != other.shape[rank_b - 2] {
            panic!("matrix not compatible");
        }

        // if a_dim=2, b_dim =2 return matmul
        if rank_a == 2 && rank_b == 2 {
            let a = self.to_ndarray();
            let b = other.to_ndarray();

            let a2d = a.into_shapeality::<ndarray::Ix2>().unwrap();
            let b2d = b.into_shapeality::<ndarray::Ix2>().unwrap();
            let c2d = a2d.dot(&b2d);
            Tensor::from_ndarray(c2d.into_dyn())
        } else {
            // create a shared shape
            let (a_batch, a_mat) = self.shape.split(rank_a - 2);
            let (b_batch, b_mat) = other.shape.split(rank_b - 2);

            // shape broadcast
            let batch_shape = Shape::union(a_batch, b_batch).unwrap();

            let mut a_shape = batch_shape;
            let mut b_shape = batch_shape;

            a_shape.extend(&a_mat);
            b_shape.extend(&b_mat);

            // real broadcast
            let a = self.upcast(a_shape).unwrap();
            let b = other.upcast(b_shape).unwrap();

            let c = a
                .along_axis(0)
                .zip(b.along_axis(0))
                .map(|(a, b)| a.matmul(&b))
                .collect::<Vec<Tensor>>();

            Tensor::stack(&c, 0).unwrap()
        }
    }

    // * Creates new tensor
    // batch operations are allowed
    pub fn matvec(&self, v: &Tensor) -> Tensor {
        // (A, B) * (B, 1) = (A, 1)
        let v = v.expand_dims(-1);
        self.matmul(&v).squeeze(-1)
    }

    ////////// Iterator////////////

    pub fn along_axis<I>(&self, axis: I) -> AlongAxisIter
        where I: ToIndex
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
        let mut folded = Tensor::from_elem(folded_shape, init);

        for t in self.along_axis(axis) {
            // wow!
            folded.random_iter_mut()
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
                    let offset_ptr = self.arr().as_ptr().offset(
                        self.offset as isize
                    );
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
                let offset_ptr = self.arr_mut().as_mut_ptr().offset(self.offset as isize);
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
        if let Ok(u) = self.shape.union(&other.shape) {
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

    // element iteration
    fn unordered_foreach(&self) {
        // only offset changes are exist
        if self.is_contiguous() {

            // transpose, reshape .... ,
        }

        // shape -> make it ordered.... and

        // if there is
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
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 12., ]),
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 12., ])
        );

        assert_ne!(
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 12., ]),
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 13., ])
        );

        assert_ne!(
            Tensor::from_slice([1, 3, 2], &[1., 4., 6., 8., 10., 12., ]),
            Tensor::from_slice([1, 1, 1, 3, 2], &[1., 4., 6., 8., 10., 12., ])
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
        assert_eq!(a.shape(), [1, 4, 10].to_shape());
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
    fn test_data() {}

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
        assert_eq!(Tensor::cat(&[a, b], 2).unwrap().shape(), [3, 2, 10].to_shape());
    }

    #[test]
    fn test_stack() {
        let a = Tensor::ones([3, 2, 5]);
        let b = Tensor::ones([3, 2, 5]);
        assert_eq!(Tensor::stack(&[a, b], 2).unwrap().shape(), [3, 2, 2, 5].to_shape());
    }

    #[test]
    fn test_squeeze() {
        let a = Tensor::ones([1, 3, 2, 1]);
        assert_eq!(a.squeeze(-1).shape(), [1, 3, 2].to_shape());
        assert_eq!(a.squeeze(0).shape(), [3, 2, 1].to_shape());
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
        assert_eq!(a.expand_dims(3).shape(), [3, 2, 9, 1].to_shape());
        assert_eq!(a.expand_dims(-1).shape(), [3, 2, 9, 1].to_shape());
        assert_eq!(a.expand_dims(0).shape(), [1, 3, 2, 9].to_shape());
        assert_eq!(a.expand_dims(1).shape(), [3, 1, 2, 9].to_shape());
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
        assert_eq!(a.upcast([3, 1, 9]).unwrap().shape(), [3, 1, 9].to_shape());
        assert_eq!(a.upcast([3, 1, 9]).unwrap(), Tensor::ones([3, 1, 9]));

        assert_eq!(a.upcast([10, 3, 7, 9]).unwrap().shape(), [10, 3, 7, 9].to_shape());
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
            .for_each(|t| assert_eq!(t.shape(), [3, 1].to_shape()));
        a.along_axis(0).for_each(|t| assert_eq!(t.shape(), [1, 9].to_shape()));
    }

    #[test]
    fn test_fold_axis() {
        let a = Tensor::ones([3, 1, 9]);

        let b = a.fold_axis(-1, 0.0, |&a, &b| a + b);

        assert_eq!(b, Tensor::from_elem([3, 1], 9.0));
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_slice([2, 2, 2], &[1., 2., 3., 4., 3., 4., 5., 6.]);

        let b = Tensor::from_slice([2, 2], &[2., 0., 0., 2.]);

        assert_eq!(
            a.matmul(&b),
            Tensor::from_slice([2, 2, 2], &[2., 4., 6., 8., 6., 8., 10., 12., ])
        );
    }

    #[test]
    fn test_matvec() {
        let a = Tensor::from_slice([2, 2, 2], &[1., 2., 3., 4., 3., 4., 5., 6.]);

        let b = Tensor::from_slice([2, 2], &[1., 2., 3., 1.]);

        assert_eq!(
            a.matvec(&b),
            Tensor::from_slice([2, 2], &[5., 11., 13., 21., ])
        );
    }

    #[test]
    fn test_mapv() {}

    #[test]
    fn test_to_unsigned_index() {
        assert_eq!(to_unsigned_index(-3, 3), 0);
        assert_eq!(to_unsigned_index(-2, 3), 1);
        assert_eq!(to_unsigned_index(-1, 3), 2);
        assert_eq!(to_unsigned_index(0, 3), 0);
        assert_eq!(to_unsigned_index(1, 3), 1);
        assert_eq!(to_unsigned_index(2, 3), 2);
        assert_eq!(to_unsigned_index(3, 3), 3);
        assert_eq!(to_unsigned_index(4, 3), 4);
        assert_eq!(to_unsigned_index(5, 3), 5);
    }
}
