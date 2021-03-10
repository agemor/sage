use crate::autodiff::var::Var;
use crate::layers::{Parameter, Stackable};

pub struct Conv2d {
    num_in_chan: usize,
    num_out_chan: usize,
    kernel_size: usize,
    stride: usize,
}

impl Conv2d {
    pub fn new(num_in_chan: usize, num_out_chan: usize, kernel_size: usize, stride: usize) -> Self {
        Conv2d {
            num_in_chan,
            num_out_chan,
            kernel_size,
            stride,
        }
    }
}

impl Parameter for Conv2d {}

impl Stackable for Conv2d {
    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }
}
