use crate::autodiff::Var;
use crate::layers::Layer;

pub struct ReLU;

impl Layer for ReLU {
    fn init(&self) {}

    fn pass(&self, x: &Var) -> Var {
        crate::op::relu(x)
    }

    fn params(&self) -> Option<Vec<&Var>> {
        None
    }
}
