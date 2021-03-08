pub mod activations;
pub mod loss;

use crate::autodiff::Var;
use crate::tensor::Tensor;
use crate::{ops, tensor};

pub trait Layer {
    fn init(&self);
    fn forward(&self, x: &Var) -> Var;
    fn params(&self) -> Option<Vec<&Var>>;
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

pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    pub fn from(layers: Vec<Box<dyn Layer>>) -> Self {
        Sequential { layers }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
}

impl Layer for Sequential {
    fn init(&self) {
        for layer in self.layers.iter() {
            layer.init();
        }
    }

    fn forward(&self, x: &Var) -> Var {
        self.layers
            .iter()
            .fold(x.clone(), |x, layer| layer.forward(&x))
    }

    fn params(&self) -> Option<Vec<&Var>> {
        let mut params = Vec::new();

        for layer in self.layers.iter() {
            if let Some(p) = layer.params() {
                params.extend(p)
            }
        }
        Some(params)
    }
}

pub struct Affine {
    pub kernel: Var,
    pub bias: Var,
}

impl Affine {
    pub fn new(input: usize, output: usize) -> Self {
        Affine {
            kernel: Var::with_shape([output, input]),
            bias: Var::with_shape([output]),
        }
    }
}

impl Layer for Affine {
    fn init(&self) {
        // do some Kaiming init (targeted for the ReLU)
        self.kernel
            .set_data(tensor::init::kaiming_uniform(self.kernel.shape(), 1.0));

        self.bias.set_data(Tensor::zeros(self.bias.shape()));
    }

    fn forward(&self, x: &Var) -> Var {
        ops::matvec(&self.kernel, x) + &self.bias
    }

    fn params(&self) -> Option<Vec<&Var>> {
        Some(vec![&self.kernel, &self.bias])
    }
}
