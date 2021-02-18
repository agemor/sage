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
use crate::tensor::shape::{Dim, IntoDimension, ShapeError};

pub mod init;
pub mod iter;
pub mod math;
pub mod shape;

/////////////////////////// Tensor implementation ///////////////////////////

pub struct Tensor {
    // keep it private

    // tensor dim
    dim: Dim,

    // array offset and strides
    strides: Vec<usize>,
    offset: usize,

    // underlying array that holds actual data
    arr: Rc<UnsafeCell<Vec<f32>>>,
}

impl Tensor {
    /////////////////////// basic functions ///////////////////////

    pub fn shape(&self) -> &[usize] {
        &self.dim.sizes
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    // number of dimensions (also known as .. order, degree, ndims)
    pub fn rank(&self) -> usize {
        self.dim.ndim()
    }

    // total number of elements in this tensor
    pub fn size(&self) -> usize {
        self.dim.size()
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
        let mut bij_dim = Dim::empty(); // 'bijective (no-broadcast)' dimension
        let mut bij_strides = Vec::new();

        for (s, d) in zip(&self.strides, &self.dim.sizes) {
            // count only 'real' dims
            if *s > 0 {
                bij_strides.push(*s);
                bij_dim.sizes.push(*d);
            }
        }
        // sort descending order (remove effects of reshaping operations)
        bij_strides.sort_by(|a, b| b.cmp(a));

        zip(bij_strides, bij_dim.default_strides()).all_equal()
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
    pub fn from_slice<D>(shape: D, s: &[f32]) -> Self
    where
        D: IntoDimension,
    {
        Tensor::from_vec(shape, s.to_vec())
    }

    pub fn from_vec<D>(shape: D, v: Vec<f32>) -> Self
    where
        D: IntoDimension,
    {
        // materialization of shape
        let dim = shape.into_dimension();
        let strides = dim.default_strides();

        // check shape-vector compatibility
        if dim.size() != v.len() {
            panic!("shape and vector not compatible")
        }

        Tensor {
            dim,
            strides,
            offset: 0,
            arr: Rc::new(UnsafeCell::new(v)),
        }
    }

    pub fn scalar(v: f32) -> Self {
        Tensor::from_elem([1], v)
    }

    // Create tensor from single element
    pub fn from_elem<D>(shape: D, elem: f32) -> Self
    where
        D: IntoDimension,
    {
        let dim = shape.into_dimension();
        let v = vec![elem; dim.size()];
        Tensor::from_vec(dim, v)
    }

    // Create tensor from the given distribution
    pub fn from_dist<D, T>(shape: D, dist: T) -> Tensor
    where
        D: IntoDimension,
        T: Distribution<f32>,
    {
        let dim = shape.into_dimension();

        let mut v = Vec::<f32>::with_capacity(dim.size());
        let mut rng = &mut SmallRng::from_rng(thread_rng()).unwrap();

        for _ in 0..dim.size() {
            v.push(dist.sample(rng));
        }

        Tensor::from_vec(dim, v)
    }

    fn view<D>(orig: &Tensor, shape: D, strides: &[usize], offset: usize) -> Tensor
    where
        D: IntoDimension,
    {
        Tensor {
            dim: shape.into_dimension(),
            strides: strides.to_vec(),
            offset,
            arr: orig.arr.clone(), // this increases reference counter (do not copy actual data)
        }
    }

    // private
    fn from_ndarray(arr: ndarray::ArrayD<f32>) -> Self {
        let dim = Dim::new(arr.shape());
        let vec = arr.into_raw_vec();
        Tensor::from_vec(dim, vec) // must unwrap
    }

    ////////////////// ndarray view converter (private) ////////////////

    fn data(&self) -> ndarray::ArrayViewD<f32> {
        unsafe {
            // let inner = self.arr.get().as_ref().unwrap();

            let mut raw_ptr = self.arr().as_ptr();
            // calculate offset
            raw_ptr = raw_ptr.offset(self.offset as isize);

            let shape = ndarray::IxDyn(self.shape()).strides(ndarray::IxDyn(self.strides()));

            ndarray::ArrayViewD::from_shape_ptr(shape, raw_ptr)
        }
    }

    ////////////////////// optimizers

    fn shrink_if_possible(&self) {
        if Rc::strong_count(&self.arr) == 1 {
            // for cases where actual_size < current_size (i.e., broadcast), we does nothing special.
            if self.mem_size() > self.size() {
                self.standalone();
            }
        }
    }

    pub fn standalone(&self) {
        let owned = self.data().to_owned();
        let new_v = owned.into_raw_vec();
        unsafe {
            *self.arr.get() = new_v;
        }
    }

    ///////////////// init constructors /////////////////

    pub fn zeros<S>(shape: S) -> Tensor
    where
        S: IntoDimension,
    {
        Tensor::from_elem(shape, 0.0)
    }

    pub fn ones<S>(shape: S) -> Tensor
    where
        S: IntoDimension,
    {
        Tensor::from_elem(shape, 1.0)
    }

    // sample from standard normal dist
    pub fn randn<S>(shape: S) -> Tensor
    where
        S: IntoDimension,
    {
        Tensor::from_dist(shape, Normal::new(0.0, 1.0).unwrap())
    }

    ///////////////// index, slice, join /////////////////
    // * Creates new tensor
    pub fn cat(t: &[Tensor], axis: usize) -> Result<Tensor, ShapeError> {
        // convert into array views
        let t_data = t
            .iter()
            .map(|t| t.data())
            .collect::<Vec<ndarray::ArrayViewD<f32>>>();

        if let Ok(arr) = ndarray::concatenate(ndarray::Axis(axis), &t_data) {
            Ok(Tensor::from_ndarray(arr))
        } else {
            Err(ShapeError::new("cannot concat!"))
        }
    }

    // * Creates new tensor
    pub fn stack(t: &[Tensor], axis: usize) -> Result<Tensor, ShapeError> {
        // convert into array views
        let t_data = t
            .iter()
            .map(|t| t.data())
            .collect::<Vec<ndarray::ArrayViewD<f32>>>();
        if let Ok(arr) = ndarray::stack(ndarray::Axis(axis), &t_data) {
            Ok(Tensor::from_ndarray(arr))
        } else {
            Err(ShapeError::new("cannot stack!"))
        }
    }

    pub fn squeeze(&self, axis: isize) -> Tensor {
        let axis = to_unsigned_index(axis, self.rank());

        let mut new_dim = self.dim.clone();
        let mut new_strides = self.strides.clone();

        if new_dim.ndim() <= axis {
            panic!("axis out of range");
        }

        if new_dim[axis] == 1 {
            new_dim.remove(axis);
            new_strides.remove(axis);
        } else {
            panic!("dim=1 cannot be squeezed");
        }
        Tensor::view(self, new_dim, &new_strides, self.offset)
    }

    pub fn expand_dims(&self, axis: isize) -> Tensor {
        // allow unexisting index
        let axis = to_unsigned_index(axis, self.rank() + 1);

        let mut new_dim = self.dim.clone();
        let mut new_strides = self.strides.clone();

        // special case!
        if new_dim.ndim() < axis {
            panic!("axis out of range");
        }

        new_dim.add(axis, 1);

        if new_strides.len() == axis {
            new_strides.push(1);
        } else {
            let s = self.strides[axis];
            new_strides.insert(axis, s);
        }

        Tensor::view(self, new_dim, &new_strides, self.offset)
    }

    // reshape (underlying data does not change)
    pub fn reshape<D>(&self, shape: D) -> Result<Tensor, ShapeError>
    where
        D: IntoDimension,
    {
        let new_dim = shape.into_dimension();
        let new_strides = new_dim.default_strides();

        if self.is_contiguous() {
            if self.size() == new_dim.size() {
                Ok(Tensor::view(self, new_dim, &new_strides, self.offset))
            } else {
                Err(ShapeError::new("shape not compatible"))
            }
        } else {
            Err(ShapeError::new("tensor not contiguous"))
        }
    }

    // swap last two dims of tensor
    pub fn transpose(&self, ax: isize, bx: isize) -> Tensor {
        let ax = to_unsigned_index(ax, self.rank());
        let bx = to_unsigned_index(bx, self.rank());

        if ax == bx {
            panic!("same axis");
        }

        if self.rank() < 2 {
            panic!("tensor too small");
        }

        let mut new_dim = self.dim.clone();
        let mut new_strides = self.strides.clone();

        new_dim.sizes.swap(ax, bx);
        new_strides.swap(ax, bx);

        Tensor::view(self, new_dim, &new_strides, self.offset)
    }

    pub fn permute(&self, axes: &[isize]) -> Tensor {
        let axes = axes
            .iter()
            .map(|axis| to_unsigned_index(*axis, self.rank()))
            .collect::<Vec<usize>>();

        if axes.iter().any(|axis| *axis >= self.rank()) {
            panic!("axis out of range");
        }

        let mut use_counts = vec![0; self.rank()];

        axes.iter().for_each(|axis| {
            use_counts[*axis] += 1;
        });

        if use_counts.iter().any(|count| *count != 1) {
            panic!("some axes are not used, or used more than once");
        }

        let mut new_dim = self.dim.clone();
        let mut new_strides = self.strides.clone();

        for axis in axes {
            new_dim[axis] = self.dim[axis];
            new_strides[axis] = self.strides[axis];
        }

        Tensor::view(self, new_dim, &new_strides, self.offset)
    }

    pub fn upcast<D>(&self, shape: D) -> Result<Tensor, ShapeError>
    where
        D: IntoDimension,
    {
        let target_dim = shape.into_dimension();

        if self.rank() > target_dim.ndim() {
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

        let mut new_dim = self.dim.clone();
        let mut new_strides = self.strides.clone();

        let pad_len = target_dim.ndim() - new_dim.ndim();

        for _ in 0..pad_len {
            new_dim.add(0, 1);
            new_strides.insert(0, 0);
        }

        for ((a, b), s) in new_dim
            .iter_mut()
            .zip(target_dim.iter())
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
        Ok(Tensor::view(self, new_dim, &new_strides, self.offset))
    }

    /// make possible a[..][3][..] kind of operations.
    pub fn index_axis(&self, index: isize, axis: isize) -> Tensor {
        let axis = to_unsigned_index(axis, self.rank());
        let index = to_unsigned_index(index, self.dim[axis]);

        if self.dim[axis] <= index {
            panic!("axis out of range");
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

        let offset = self.strides[axis] * index;

        let mut new_dim = self.dim.clone();
        let mut new_stride = self.strides.clone();

        new_dim.remove(axis);

        new_stride.remove(axis);

        Tensor::view(self, new_dim, &new_stride, self.offset + offset)
    }

    // single index.. unwraps first one.
    pub fn index(&self, index: isize) -> Tensor {
        self.index_axis(index, 0)
    }

    // make possible of a[..][2..3][..] kind of operations
    pub fn slice_axis(&self, start_index: isize, end_index: isize, axis: isize) -> Tensor {
        let axis = to_unsigned_index(axis, self.rank());

        let stride = self.strides[axis];

        let dim_size = self.dim[axis];

        let start_index = to_unsigned_index(start_index, dim_size);
        let end_index = to_unsigned_index(end_index, dim_size);

        if start_index > end_index {
            panic!("start and end index are not in the order");
        }

        if dim_size <= end_index {
            panic!("axis range overflow");
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

        let offset = stride * start_index;

        let mut new_dim = self.dim.clone();
        let mut new_stride = self.strides.clone();

        new_dim[axis] = end_index - start_index + 1;
        new_stride.remove(axis);

        Tensor::view(self, new_dim, &new_stride, self.offset + offset)
    }

    pub fn slice(&self, start_index: isize, end_index: isize) -> Tensor {
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
        if self.dim[rank_a - 1] != other.dim[rank_b - 2] {
            panic!("matrix not compatible");
        }

        // if a_dim=2, b_dim =2 return matmul
        if rank_a == 2 && rank_b == 2 {
            // USES NDARRAY

            let a = self.data();
            let b = other.data();

            let a2d = a.into_dimensionality::<ndarray::Ix2>().unwrap();
            let b2d = b.into_dimensionality::<ndarray::Ix2>().unwrap();
            let c2d = a2d.dot(&b2d);
            Tensor::from_ndarray(c2d.into_dyn())
        } else {
            // create a shared shape
            let (a_bat_dim, a_mat_dim) = self.dim.split(rank_a - 2);
            let (b_bat_dim, b_mat_dim) = other.dim.split(rank_b - 2);

            // shape broadcast
            let bat_dim = a_bat_dim.union(&b_bat_dim).unwrap();

            let mut a_dim = bat_dim.clone();
            let mut b_dim = bat_dim.clone();

            a_dim.extend(&a_mat_dim);
            b_dim.extend(&b_mat_dim);

            // real broadcast
            let a = self.upcast(a_dim).unwrap();
            let b = other.upcast(b_dim).unwrap();

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

    pub fn along_axis(&self, axis: isize) -> AlongAxisIter {
        let axis = to_unsigned_index(axis, self.rank());
        if self.rank() <= axis {
            panic!("axis out of range");
        }
        AlongAxisIter::new(self, axis)
    }

    pub fn fold_axis<F>(&self, axis: isize, init: f32, mut fold: F) -> Tensor
    where
        F: FnMut(&f32, &f32) -> f32,
    {
        let axis_u = to_unsigned_index(axis, self.rank());

        let mut res_dim = self.dim.clone();
        res_dim.remove(axis_u);
        let mut res = Tensor::from_elem(res_dim, init);

        for t in self.along_axis(axis) {
            // wow!
            res.random_iter_mut()
                .zip(t.logical_iter())
                .for_each(|(x, y)| {
                    *x = fold(x, y);
                })
        }
        res
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
                s.iter_mut().map(|x| *x = f(x));
            }
        }
        // order does not matter..  every element is only visited once.
        else {
            self.random_iter_mut().map(|x| *x = f(x));
        }
    }

    // visits each elements in a logical order.
    // used for pairwise tensor operations.
    pub fn logical_iter(&self) -> Iter {
        Iter::new(
            self.arr().as_ptr(),
            self.offset,
            self.shape(),
            self.strides(),
        )
    }

    // Visits inner elements in an arbitrary order, but faster than logical iterator
    // because it visits elements in a way that cpu cache is utilized.
    pub fn random_iter_mut(&self) -> IterMut {
        let (strides, dim): (Vec<usize>, Vec<usize>) = self
            .strides
            .iter()
            .zip(self.dim.iter())
            .filter(|(&s, _)| s > 0)
            .sorted_by(|(&s1, _), (&s2, _)| s2.cmp(&s1))
            .unzip();

        IterMut::new(self.arr_mut().as_mut_ptr(), self.offset, &dim, &strides)
    }

    pub fn zip_map<F>(&self, other: &Tensor, f: F) -> Result<Tensor, ShapeError>
    where
        F: Fn(&f32, &f32) -> f32,
    {
        if let Ok(u) = self.dim.union(&other.dim) {
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

pub(crate) fn to_unsigned_index(index: isize, bound: usize) -> usize {
    if index >= 0 {
        // allows index oor
        index as usize
    } else {
        // does not allow oor
        index.rem_euclid(bound as isize) as usize
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
        let v = self.logical_iter().map(|x| *x).collect::<Vec<f32>>();
        Tensor::from_vec(self.shape(), v)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        //writeln!(f, "{:?}", unsafe { self.inner.get().as_ref() }.unwrap());
        writeln!(f, "{}", self.data())
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("dim", &self.dim)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("arr\n", &self.data())
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
                &[1., 4., 6., 8., 10., 12., 8., 10., 12., 14., 16., 19.]
            )
        );

        assert_eq!(
            c,
            Tensor::from_slice(
                [2, 3, 2],
                &[2., 4., 6., 8., 10., 12., 8., 10., 12., 14., 16., 18.]
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
        assert_eq!(a.shape(), &[1, 4, 10]);
        assert_eq!(a.dim[0], 1);
        assert_eq!(a.dim[1], 4);
        assert_eq!(a.dim[2], 10);
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
        assert_eq!(Tensor::cat(&[a, b], 2).unwrap().shape(), &[3, 2, 10]);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::ones([3, 2, 5]);
        let b = Tensor::ones([3, 2, 5]);
        assert_eq!(Tensor::stack(&[a, b], 2).unwrap().shape(), &[3, 2, 2, 5]);
    }

    #[test]
    fn test_squeeze() {
        let a = Tensor::ones([1, 3, 2, 1]);
        assert_eq!(a.squeeze(-1).shape(), &[1, 3, 2]);
        assert_eq!(a.squeeze(0).shape(), &[3, 2, 1]);
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
        assert_eq!(a.expand_dims(3).shape(), &[3, 2, 9, 1]);
        assert_eq!(a.expand_dims(-1).shape(), &[3, 2, 9, 1]);
        assert_eq!(a.expand_dims(0).shape(), &[1, 3, 2, 9]);
        assert_eq!(a.expand_dims(1).shape(), &[3, 1, 2, 9]);
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
        assert_eq!(a.upcast([3, 1, 9]).unwrap().shape(), &[3, 1, 9]);
        assert_eq!(a.upcast([3, 1, 9]).unwrap(), Tensor::ones([3, 1, 9]));

        assert_eq!(a.upcast([10, 3, 7, 9]).unwrap().shape(), &[10, 3, 7, 9]);
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
            .for_each(|t| assert_eq!(t.shape(), &[3, 1]));
        a.along_axis(0).for_each(|t| assert_eq!(t.shape(), &[1, 9]));
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
            Tensor::from_slice([2, 2, 2], &[2., 4., 6., 8., 6., 8., 10., 12.,])
        );
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
