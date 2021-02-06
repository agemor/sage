use crate::autodiff::Var;
use std::collections::HashMap;

pub trait Optimizer {
    fn init(&self) {}

    fn update(&self, grads: &HashMap<&Var, Var>);
}

pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> SGD {
        SGD { lr }
    }
}

impl Optimizer for SGD {

    fn update(&self, grads: &HashMap<&Var, Var>) {
        for (param, grad) in grads {}
    }
}
