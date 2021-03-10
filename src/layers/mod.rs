use crate::autodiff::var::Var;

pub mod activations;
pub mod attention;
pub mod base;
pub mod convolutional;
pub mod embedding;
pub mod normalization;
pub mod pooling;
pub mod recurrent;

pub trait Parameter {
    fn init(&self) {}
    fn params(&self) -> Option<Vec<&Var>> {
        None
    }
}

pub trait Stackable: Parameter {
    fn forward(&self, x: &Var) -> Var;
}

pub fn gather_params(params: Vec<Option<Vec<&Var>>>) -> Option<Vec<&Var>> {
    let mut res = Vec::<&Var>::new();
    for p in params {
        if let Some(param) = p {
            res.extend(param);
        }
    }
    if res.is_empty() {
        None
    } else {
        Some(res)
    }
}
