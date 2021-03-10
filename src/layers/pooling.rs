use crate::autodiff::var::Var;
use crate::layers::{Parameter, Stackable};

pub struct AvgPool2d {
    kernel_size: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize) -> Self {
        AvgPool2d { kernel_size }
    }
}

impl Parameter for AvgPool2d {}

impl Stackable for AvgPool2d {
    fn forward(&self, x: &Var) -> Var {
        unimplemented!()
    }
}
