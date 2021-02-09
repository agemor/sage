#![feature(duration_constants)]
#![feature(duration_zero)]
#![feature(box_syntax)]
#[macro_use]
extern crate impl_ops;

use crate::autodiff::diff;
use crate::layers::activations::ReLU;
use crate::layers::{Affine, Layer, Sequential};
use crate::optimizers::Optimizer;
use crate::mnist::Mnist;
use crate::data::Dataset;

mod autodiff;
mod layers;
mod net;
mod op;
mod optimizers;
mod tensor;
mod utils;
mod data;
mod mnist;

fn main() {

    // Load dataset
    let mnist = Mnist::from_source(
        "./data/train-images.idx3-ubyte",
        "./data/train-labels.idx1-ubyte",
    ).unwrap();

    // Model
    let model = Sequential::from(vec![
        box Affine::new(256, 32),
        box ReLU,
        box Affine::new(32, 16),
        box ReLU,
    ]);

    // Optimizer
    let optimizer = optimizers::SGD::new(0.0001);

    // Initialize model weights and optimizer params
    model.init();
    optimizer.init();

    for (images, labels) in mnist.iter().batch(10, Mnist::collate){
        println!("{}", i);


        let logits = model.pass(&input);
        let loss = op::softmax_cross_entropy(&logits, &label);

        let params = model.params().unwrap();
        let grads = diff(&loss, &params);

        optimizer.update(&grads);
    }




    // * Inline gradient update *
    // input.set_data();
    // label.set_data();
    // op::recompute(&grads); // super-fast
    // optimizer.update(&grads);

    // * Evaluating higher-order derivatives *


    println!("Hello, world!");
}
