use crate::autodiff::var::Var;
use crate::layers::{Parameter, Stackable};

pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize) -> Self {
        AvgPool2d {
            kernel_size,
            stride: kernel_size,
            padding: 0,
        }
    }
}

impl Parameter for AvgPool2d {}

impl Stackable for AvgPool2d {
    fn forward(&self, x: &Var) -> Var {
        // N, C, H, W = x.shape
        // KH, KW = pair(kernel_size)
        // PH, PW = pair(pad)
        // SH, SW = pair(stride)
        // OH = get_conv_outsize(H, KH, SH, PH)
        // OW = get_conv_outsize(W, KW, SW, PW)
        //
        // col = im2col(x, kernel_size, stride, pad, to_matrix=True)
        // col = col.reshape(-1, KH * KW)
        // y = col.max(axis=1)
        // y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

        let batch_size = x.shape()[0];
        let in_channels = x.shape()[1];
        // (N, C, KH, KW, OH, OW)
        let col = x.im2col(self.kernel_size, self.stride, self.padding, 1);

        let col_h = col.shape()[4];
        let col_w = col.shape()[5];

        // (N*C*OH*OW, KH*KW)
        let col = col
            .permute([0, 1, 4, 5, 2, 3])
            .reshape([0, self.kernel_size * self.kernel_size]);

        let col_mean = col.mean(1, false);

        let y = col_mean
            .reshape([batch_size, col_h, col_w, in_channels])
            .permute([0, 3, 1, 2]);

        //println!("x: {}      y: {}", x.shape(), y.shape());

        y
    }
}
