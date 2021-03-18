use crate::autodiff::ops::conv::get_deconv_size;
use crate::autodiff::var::Var;
use crate::layers::base::Dense;
use crate::layers::{Parameter, Stackable};
use crate::tensor;
use crate::tensor::Tensor;

pub struct Conv2d {
    filter: Var,
    bias: Var,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Conv2d {
            filter: Var::with_shape([in_channels * kernel_size * kernel_size, out_channels]),
            bias: Var::with_shape([out_channels]),
            kernel_size,
            stride,
            padding,
            dilation: 1,
        }
    }
}

impl Parameter for Conv2d {
    fn init(&self) {
        self.filter.set_data(Tensor::null());

        //self.filter
        //    .set_data(tensor::init::kaiming_uniform(self.filter.shape(), 1.0));
        self.bias.set_data(Tensor::zeros(self.bias.shape()));
    }
    fn params(&self) -> Option<Vec<&Var>> {
        Some(vec![&self.filter, &self.bias])
    }
}

impl Stackable for Conv2d {
    fn forward(&self, x: &Var) -> Var {
        let batch_size = x.shape()[0];

        //println!("{}          kernel_size: {}, stride: {}, padding: {}", self.filter.shape(), self.kernel_size, self.stride, self.padding);

        let col = x.im2col(self.kernel_size, self.stride, self.padding, self.dilation);

        let col_h = col.shape()[4];
        let col_w = col.shape()[5];

        // (N*OH*OW, C*KH*KW)
        let col = col
            .permute([0, 4, 5, 1, 2, 3])
            .reshape([batch_size * col_h * col_w, 0]);

        let out = col.matmul(&self.filter) + &self.bias;

        let y = out
            .reshape([batch_size, col_h, col_w, 0])
            .permute([0, 3, 1, 2]);

        // println!("{} -> {}          kernel_size: {}, stride: {}, padding: {}", x.shape(), y.shape(), self.kernel_size, self.stride, self.padding);

        y
    }
}

pub struct TransposedConv2d {
    filter: Var,
    bias: Var,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl TransposedConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        TransposedConv2d {
            filter: Var::with_shape([in_channels, out_channels * kernel_size * kernel_size]),
            bias: Var::with_shape([out_channels * kernel_size * kernel_size]),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation: 1,
        }
    }
}

impl Parameter for TransposedConv2d {
    fn init(&self) {
        self.filter.set_data(Tensor::null());

        //self.filter
        //    .set_data(tensor::init::kaiming_uniform(self.filter.shape(), 1.0));
        self.bias.set_data(Tensor::zeros(self.bias.shape()));
    }
    fn params(&self) -> Option<Vec<&Var>> {
        Some(vec![&self.filter, &self.bias])
    }
}

impl Stackable for TransposedConv2d {
    fn forward(&self, x: &Var) -> Var {
        //
        // Weight = W
        // SH, SW = self.stride
        // PH, PW = self.pad
        // C, OC, KH, KW = Weight.shape
        // N, C, H, W = x.shape
        // if self.outsize is None:
        //     out_h = get_deconv_outsize(H, KH, SH, PH)
        //     out_w = get_deconv_outsize(W, KW, SW, PW)
        // else:
        //     out_h, out_w = pair(self.outsize)
        // img_shape = (N, OC, out_h, out_w)
        //
        // gcol = xp.tensordot(Weight, x, (0, 1))
        // gcol = xp.rollaxis(gcol, 3)
        // y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
        //                  to_matrix=False)
        // # b, k, h, w
        // if b is not None:
        //     self.no_bias = True
        //     y += b.reshape((1, b.size, 1, 1))
        // return y

        let batch_size = x.shape()[0];
        let in_channels = x.shape()[1];
        let img_h = x.shape()[2];
        let img_w = x.shape()[3];

        let out_img_h = get_deconv_size(img_h, self.kernel_size, self.stride, self.padding, 1);
        let out_img_w = get_deconv_size(img_w, self.kernel_size, self.stride, self.padding, 1);

        //println!("{}          kernel_size: {}, stride: {}, padding: {}", self.filter.shape(), self.kernel_size, self.stride, self.padding);

        // (N, C, KH, KW, OH, OW)

        // (C, O, KH, KW) * (N, C, H, W) -> (N, O, KH, KW, OH, OW)

        // (NHW, C) * (C, OKK) -> (NHW, OKK) -> (N, H, W, O, K, K)

        // (N, C, H, W) -> (NHW, C)
        let img = x.permute([0, 2, 3, 1]).reshape([0, in_channels]);

        // (NHW, C) * (C, OKK)  -> (NHW, OKK)
        let t_conv = img.matmul(&self.filter) + &self.bias;

        //  (NHW, OKK) -> (N, H, W, O, K, K) -> (N, O, K, K, H, W)
        let col = t_conv
            .reshape([
                batch_size,
                img_h,
                img_w,
                self.out_channels,
                self.kernel_size,
                self.kernel_size,
            ])
            .permute([0, 3, 4, 5, 1, 2]);

        // (N, C, KH, KW, OH, OW) -> (N, C, H, W)
        let img = col.col2im(
            out_img_w,
            out_img_h,
            self.kernel_size,
            self.stride,
            self.padding,
            1,
        );

        img
    }
}
