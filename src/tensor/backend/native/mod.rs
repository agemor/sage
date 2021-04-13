use crate::tensor::{backend, Tensor};

use crate::tensor::backend::native::iter::BufferIndexIter;
use crate::tensor::backend::{BinaryIndexOperation, BinaryOperation, UnaryOperation};
use crate::tensor::shape::ToShape;
use itertools::Itertools;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::cell::RefCell;
use std::cmp::Ordering;

mod iter;

#[derive(Clone)]
pub struct Backend;

impl Backend {
    pub fn new() -> Self {
        Backend
    }

    pub fn unary_op_fn<F>(&self, input: &Tensor, output: &Tensor, f: F)
    where
        F: Fn(f32) -> f32,
    {
        let data_in = input.buffer.as_native().data.borrow();
        let data_out = output.buffer.as_native().data.borrow_mut();

        let iter_in = BufferIndexIter::from_tensor(input);
        let iter_out = BufferIndexIter::from_tensor(output);

        iter_out
            .zip(iter_in)
            .par_bridge()
            .for_each(|(out, in_)| data_out[out] = f(data_in[in_]));
    }

    pub fn binary_op_fn<F>(&self, input1: &Tensor, input2: &Tensor, output: &Tensor, f: F)
    where
        F: Fn(f32, f32) -> f32,
    {
        assert_eq!(input1.shape, input2.shape);
        assert_eq!(input1.shape, output.shape);

        let buffer_in1 = Buffer::downcast(&input1.buffer);
        let buffer_in2 = Buffer::downcast(&input2.buffer);
        let buffer_out = Buffer::downcast(&output.buffer);

        let data_in1 = buffer_in1.data.get();
        let data_in2 = buffer_in2.data.get();
        let data_out = buffer_out.data.get();

        let iter_in1 = BufferIndexIter::from_tensor(input1);
        let iter_in2 = BufferIndexIter::from_tensor(input2);
        let iter_out = BufferIndexIter::from_tensor(output);

        iter_out
            .zip(iter_in1.zip(iter_in2))
            .par_bridge()
            .for_each(|(out, (in1, in2))| data_out[out] = f(data_in1[in1], data_in2[in2]));
    }

    pub fn reduction_index_fn<F>(&self, input: &Tensor, output: &mut Tensor, f: F, axis: usize)
    where
        F: Fn(f32, f32) -> Ordering,
    {
        // reduced shape & stride
        let shape_red = input.shape[axis];
        let stride_red = input.strides[axis];

        let shape_pre = input.shape.removed(axis);

        let mut stride_pre = input.strides.clone();
        stride_pre.remove(axis);

        assert_eq!(shape_pre.to_shape(0), output.shape);

        let buffer_in = Buffer::downcast(&input.buffer);
        let buffer_out = Buffer::downcast(&output.buffer);

        let data_in = buffer_in.data.get();
        let data_out = buffer_out.data.get();

        let iter_pre = BufferIndexIter::new(shape_pre.sizes(), &stride_pre, input.offset);
        let iter_out = BufferIndexIter::from_tensor(output);

        iter_out
            .zip(iter_pre)
            .par_bridge()
            .for_each(|(out, in_pre)| {
                let iter_red = BufferIndexIter::new(&[shape_red], &[stride_red], 0);
                let (i_sel, _) = iter_red
                    .enumerate()
                    .map(|(i, in_red)| (i, data_in[in_pre + in_red]))
                    .max_by(|(i_sel, val_sel), (i, val)| f(val_sel, val))
                    .unwrap();
                data_out[out] = i_sel;
            });
    }

    pub fn reduction_fn<F>(&self, input: &Tensor, output: &mut Tensor, f: F, axes: Vec<usize>)
    where
        F: Fn(f32, f32) -> f32,
    {
        // get preserved (shape, stride) and reduced (shape, stride) pair
        let axes_pre = (0..input.order())
            .filter(|a| !axes.contains(a))
            .collect::<Vec<usize>>();

        // reduced shape & stride
        let shape_red = axes.iter().map(|&a| input.shape[a]).collect::<Vec<usize>>();
        let stride_red = axes
            .iter()
            .map(|&a| input.strides[a])
            .collect::<Vec<usize>>();

        // preserved ones
        let shape_pre = axes_pre
            .iter()
            .map(|&a| input.shape[a])
            .collect::<Vec<usize>>();

        let stride_pre = axes_pre
            .iter()
            .map(|&a| input.strides[a])
            .collect::<Vec<usize>>();

        assert_eq!(shape_pre.to_shape(0), output.shape);

        let buffer_in = Buffer::downcast(&input.buffer);
        let buffer_out = Buffer::downcast(&output.buffer);

        let data_in = buffer_in.data.get();
        let data_out = buffer_out.data.get();

        let iter_pre = BufferIndexIter::new(&shape_pre, &stride_pre, input.offset);
        let iter_out = BufferIndexIter::from_tensor(output);

        iter_out
            .zip(iter_pre)
            .par_bridge()
            .for_each(|(out, in_pre)| {
                let iter_red = BufferIndexIter::new(&shape_red, &stride_red, 0);
                data_out[out] = iter_red
                    .map(|in_red| data_in[in_pre + in_red])
                    .fold1(|acc, v| f(acc, v))
                    .unwrap();
            });
    }
}

impl backend::Backend for Backend {
    fn alloc_mem(&self, size: usize) -> backend::Buffer {
        backend::Buffer::Native(Buffer::new(size, self.clone()))
    }

    fn alloc_mem_from_iter<I>(&self, data: I) -> backend::Buffer
    where
        I: ExactSizeIterator<Item = f32>,
    {
        backend::Buffer::Native(Buffer::from_iter(data, self.clone()))
    }

    fn copy(&self, input: &Tensor, output: &mut Tensor) {
        assert_eq!(input.extents(), output.extents());

        let buffer_in = Buffer::downcast(&input.buffer);
        let mut buffer_out = Buffer::downcast(&output.buffer);

        let data_in = buffer_in.data.borrow();
        let data_out = buffer_out.data.get_mut();

        let iter_in = BufferIndexIter::from_tensor(input);
        let iter_out = BufferIndexIter::from_tensor(output);

        iter_out.zip(iter_in).par_bridge().for_each(|out, inn| {
            data_out[out] = data_in[inn];
        });
    }

    fn concat(&self, inputs: &[&Tensor], output: &mut Tensor, axis: usize) {
        if !inputs
            .iter()
            .chain([output].iter())
            .map(|t| t.shape.removed(axis))
            .all_equal()
        {
            panic!("does not match shape");
        }

        if inputs.iter().map(|t| t.shape[axis]).sum() != output.shape[axis] {
            panic!("invalid output shape");
        }

        let mut buffer_out = Buffer::downcast(&output.buffer);
        let data_out = buffer_out.data.get_mut();

        let mut k = 0;
        for &t in inputs {
            let buffer_in = Buffer::downcast(&input1.buffer);
            let data_in = buffer_in.data.borrow();
            let output_slice = output.slice_axis(k, t.shape[axis], axis);
            k += t.shape[axis];

            let iter_in = BufferIndexIter::from_tensor(t);
            let iter_out = BufferIndexIter::from_tensor(&output_slice);

            iter_out.zip(iter_in).par_bridge().for_each(|(out, in1)| {
                data_out[out] = data_in[in1];
            });
        }
    }

    fn unary_op(&self, input: &Tensor, output: &mut Tensor, op: UnaryOperation) {
        let op = match op {
            UnaryOperation::Neg => |a| -a,
            UnaryOperation::Exp => |a| a.exp(),
            UnaryOperation::Sign => |a| a.sgn(),
            UnaryOperation::Sqrt => |a| a.sqrt(),
        };

        self.unary_op_fn(input, output, op);
    }

    fn binary_op(
        &self,
        input1: &Tensor,
        input2: &Tensor,
        output: &mut Tensor,
        op: BinaryOperation,
    ) {
        let op = match op {
            BinaryOperation::Add => |a, b| a + b,
            BinaryOperation::Sub => |a, b| a - b,
            BinaryOperation::Mul => |a, b| a * b,
            BinaryOperation::Div => |a, b| a / b,
            BinaryOperation::Eq => |a, b| if a - b < 0.001 { 1.0 } else { 0.0 },
        };

        self.binary_op_fn(input1, input2, output, op);
    }

    fn reduction(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        op: BinaryOperation,
        axes: Vec<usize>,
    ) {
        let op = match op {
            BinaryOperation::Add => |a, b| a + b,
            BinaryOperation::Sub => |a, b| a - b,
            BinaryOperation::Mul => |a, b| a * b,
            BinaryOperation::Div => |a, b| a / b,
            BinaryOperation::Eq => |a, b| if a - b < 0.001 { 1.0 } else { 0.0 },
        };
        self.reduction_fn(input, output, op, axes);
    }

    fn reduction_index(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        op: BinaryIndexOperation,
        axis: usize,
    ) {
        let op = match op {
            BinaryIndexOperation::Max => |a, b| a.cmp(b),
            BinaryIndexOperation::Min => |a, b| b.cmp(a),
        };

        self.reduction_index_fn(input, output, op, axis);
    }

    fn contraction(
        &self,
        input1: &Tensor,
        input2: &Tensor,
        output: &mut Tensor,
        axes1: Vec<usize>,
        axes2: Vec<usize>,
    ) {
        // get preserved (shape, stride) and reduced (shape, stride) pair

        let axes_pre1 = (0..input1.order())
            .filter(|a| !axes1.contains(a))
            .collect::<Vec<usize>>();

        let axes_pre2 = (0..input2.order())
            .filter(|a| !axes2.contains(a))
            .collect::<Vec<usize>>();

        // reduced shape & stride
        let shape_red1 = axes1
            .iter()
            .map(|&a| input1.shape[a])
            .collect::<Vec<usize>>();
        let stride_red1 = axes1
            .iter()
            .map(|&a| input1.strides[a])
            .collect::<Vec<usize>>();

        let shape_red2 = axes2
            .iter()
            .map(|&a| input2.shape[a])
            .collect::<Vec<usize>>();
        let stride_red2 = axes2
            .iter()
            .map(|&a| input2.strides[a])
            .collect::<Vec<usize>>();

        // reduced shape must be the same
        assert_eq!(shape_red1, shape_red2);

        // preserved ones
        let shape_pre1 = axes_pre1
            .iter()
            .map(|&a| input1.shape[a])
            .collect::<Vec<usize>>();
        let stride_pre1 = axes_pre1
            .iter()
            .map(|&a| input1.strides[a])
            .collect::<Vec<usize>>();

        let shape_pre2 = axes_pre2
            .iter()
            .map(|&a| input1.shape[a])
            .collect::<Vec<usize>>();
        let stride_pre2 = axes_pre2
            .iter()
            .map(|&a| input1.strides[a])
            .collect::<Vec<usize>>();

        // output shape = shape_pre1 + shape_pre2
        let mut expected_output = shape_pre1.clone();
        expected_output.extend(&shape_pre2);
        assert_eq!(expected_output.to_shape(0), output.shape);

        let buffer_in1 = Buffer::downcast(&input1.buffer);
        let buffer_in2 = Buffer::downcast(&input2.buffer);
        let buffer_out = Buffer::downcast(&output.buffer);

        let data_in1 = buffer_in1.data.get();
        let data_in2 = buffer_in2.data.get();
        let data_out = buffer_out.data.get();

        let iter_pre1 = BufferIndexIter::new(&shape_pre1, &stride_pre1, input1.offset);
        let iter_out = BufferIndexIter::from_tensor(output);

        iter_out
            .zip(iter_pre1.flat_map(|in_base1| {
                let iter_pre2 = BufferIndexIter::new(&shape_pre2, &stride_pre2, input2.offset);
                iter_pre2.map(move |in_base2| (in_base1, in_base2))
            }))
            .par_bridge()
            .for_each(|(out, (in_base1, in_base2))| {
                let iter_red1 = BufferIndexIter::new(&shape_red1, &stride_red1, 0);
                let iter_red2 = BufferIndexIter::new(&shape_red2, &stride_red2, 0);

                let mut sum = 0.0;

                for (in1, in2) in iter_red1.zip(iter_red2) {
                    sum += data_in1[in_base1 + in1] * data_in2[in_base2 + in2];
                }
                // must unwrap
                data_out[out] = sum;
            });
    }

    fn im2col(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) {
        // (width, height)

        let batch_size = input.shape[0];
        let num_channels = input.shape[1];
        let input_size = (input.shape[3], input.shape[2]);

        let output_size = (
            (input_size.0 + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1,
            (input_size.1 + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1,
        );

        let input_len = input_size.0 * input_size.1;
        let output_len = output_size.0 * output_size.1;
        let kernel_len = kernel_size.0 * kernel_size.1;

        assert_eq!(output_size.0, output.shape[3]);
        assert_eq!(output_size.1, output.shape[2]);

        let buffer_in = Buffer::downcast(&input.buffer);
        let buffer_out = Buffer::downcast(&output.buffer);

        let data_in = buffer_in.data.get();
        let data_out = buffer_out.data.get();

        let mut index_in_offset = input.offset;
        let mut index_out_offset = output.offset;

        for n in 0..batch_size {
            index_in_offset += n * num_channels * input_len;
            index_out_offset += n * num_channels * output_len * kernel_len;

            for c in 0..num_channels {
                index_in_offset += c * input_len;
                index_out_offset += c * output_len * kernel_len;

                for h in 0..output_size.1 {
                    index_out_offset += h * output_size.0 * kernel_len;

                    for w in 0..output_size.0 {
                        index_out_offset += w * kernel_len;

                        let input_x_offset = w * stride.0 - padding.0;
                        let input_y_offset = h * stride.1 - padding.1;

                        for kh in 0..kernel_size.1 {
                            for kw in 0..kernel_size.0 {
                                let input_x = kw * dilation.0 + input_x_offset;
                                let input_y = kh * dilation.1 + input_y_offset;

                                let mut val = 0.0;
                                if input_x >= 0
                                    && input_y >= 0
                                    && input_x < input_size.0
                                    && input_y < input_size.1
                                {
                                    let index_in =
                                        input_y * input_size.0 * input_x + index_in_offset;
                                    val = data_in[index_in];
                                }
                                let index_out = kh * kernel_size.0 + kw + index_out_offset;
                                data_out[index_out] = val;
                            }
                        }
                    }
                }
            }
        }
    }

    fn col2im(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) {
        // (width, height)

        let batch_size = input.shape[0];
        let num_channels = input.shape[1];
        let input_size = (input.shape[3], input.shape[2]);

        // TODO: check expeced output size

        let output_size = (output.shape[3], output.shape[2]);

        let input_len = input_size.0 * input_size.1;
        let output_len = output_size.0 * output_size.1;
        let kernel_len = kernel_size.0 * kernel_size.1;

        let buffer_in = Buffer::downcast(&input.buffer);
        let buffer_out = Buffer::downcast(&output.buffer);

        let data_in = buffer_in.data.get();
        let data_out = buffer_out.data.get();

        let mut index_in_offset = output.offset;
        let mut index_out_offset = input.offset;

        for n in 0..batch_size {
            index_in_offset += n * num_channels * output_len * kernel_len;
            index_out_offset += n * num_channels * input_len;

            for c in 0..num_channels {
                index_in_offset += c * output_len * kernel_len;
                index_out_offset += c * input_len;

                for h in 0..output_size.1 {
                    index_in_offset += h * output_size.0 * kernel_len;

                    for w in 0..output_size.0 {
                        index_in_offset += w * kernel_len;

                        let input_x_offset = w * stride.0 - padding.0;
                        let input_y_offset = h * stride.1 - padding.1;

                        for kh in 0..kernel_size.1 {
                            for kw in 0..kernel_size.0 {
                                let output_x = kw * dilation.0 + input_x_offset;
                                let output_y = kh * dilation.1 + input_y_offset;

                                let index_in = kh * kernel_size.0 + kw + index_in_offset;

                                if output_x >= 0
                                    && output_y >= 0
                                    && output_x < input_size.0
                                    && output_y < input_size.1
                                {
                                    let index_out =
                                        output_y * input_size.0 * output_x + index_out_offset;
                                    data_out[index_out] = data_in[index_in];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct Buffer {
    backend: Backend,
    data: RefCell<Vec<f32>>,
}

impl Buffer {
    pub fn new(size: usize, backend: Backend) -> Self {
        let mut v = Vec::with_capacity(size);

        // uninitialized buffer
        unsafe {
            v.set_len(size);
        }
        Buffer {
            backend,
            data: RefCell::new(v),
        }
    }

    pub fn from_iter<I>(data: I, backend: Backend) -> Self
    where
        I: ExactSizeIterator<Item = f32>,
    {
        Buffer {
            backend,
            data: RefCell::new(data.collect_vec()),
        }
    }

    pub fn backend(&self) -> &dyn backend::Backend {
        &self.backend
    }
}
