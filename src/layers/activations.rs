use crate::autodiff::Var;
use crate::layers::Layer;

pub struct Relu;

impl Layer for Relu {
    fn init(&self) {}

    fn forward(&self, x: &Var) -> Var {
        crate::ops::relu(x)
    }

    fn params(&self) -> Option<Vec<&Var>> {
        None
    }
}
