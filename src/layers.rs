pub mod activations;
pub mod loss;

use crate::autodiff::Var;

pub trait Layer {
    fn init(&self);
    fn pass(&self, x: &Var) -> Var;
    fn params(&self) -> Option<Vec<&Var>>;
}

pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential { layers: Vec::new() }
    }

    pub fn from(layers: Vec<Box<dyn Layer>>) -> Sequential {
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

    fn pass(&self, x: &Var) -> Var {
        self.layers
            .iter()
            .fold(x.clone(), |x, layer| layer.pass(&x))
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
            kernel: Var::new(),
            bias: Var::new(),
        }
    }
}

impl Layer for Affine {
    fn init(&self) {}

    fn pass(&self, x: &Var) -> Var {
        &self.kernel * x + &self.bias
    }

    fn params(&self) -> Option<Vec<&Var>> {
        Some(vec![&self.kernel, &self.bias])
    }
}
