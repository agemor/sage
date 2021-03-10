use crate::autodiff::ops::activations::relu;
use crate::autodiff::var::Var;
use crate::layers::{Parameter, Stackable};

pub struct Relu;

impl Parameter for Relu {}
impl Stackable for Relu {
    fn forward(&self, x: &Var) -> Var {
        relu(x)
    }
}
