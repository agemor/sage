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
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        Conv2d {
            filter: Var::with_shape([in_channels * kernel_size * kernel_size, out_channels]),
            bias: Var::with_shape([out_channels]),
            kernel_size,
            stride,
            padding: 1,
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

        y
    }
}
