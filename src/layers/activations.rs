use crate::autodiff::Var;
use crate::layers::Layer;

pub struct ReLU;

impl Layer for ReLU {
    fn init(&self) {}

    fn forward(&self, x: &Var) -> Var {
        crate::ops::relu(x)
    }

    fn params(&self) -> Option<Vec<&Var>> {
        None
    }
}
