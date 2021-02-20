use crate::autodiff::Var;

use std::collections::HashMap;
use std::ops::Deref;

pub trait Optimizer {
    fn init(&self) {}

    fn update(&self, grads: HashMap<Var, Var>);
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
    fn update(&self, grads: HashMap<Var, Var>) {

        for (param, grad) in grads {
            let grad_tensor = grad.data();

            let b = grad_tensor.deref();


            param.update_grads(b * self.lr);
        }
    }
}
