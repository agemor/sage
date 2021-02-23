#![feature(duration_constants)]
#![feature(duration_zero)]
#![feature(box_syntax)]
#![feature(nll)]
#![feature(hash_drain_filter)]
#![feature(is_sorted)]
#![feature(const_generics)]
#![feature(array_methods)]
#![feature(drain_filter)]
#![feature(array_map)]

#[macro_use]
extern crate impl_ops;

use crate::autodiff::{diff, Var};
use crate::data::Dataset;
use crate::layers::activations::Relu;
use crate::layers::{Affine, Layer, Sequential};
use crate::mnist::Mnist;
use crate::optimizers::Optimizer;

mod autodiff;
mod data;
mod layers;
mod mnist;
mod ops;
mod optimizers;
mod tensor;
mod session;
mod shape;

fn main() {
    // Load dataset
    let mnist = Mnist::from_source(
        "./data/train-images.idx3-ubyte",
        "./data/train-labels.idx1-ubyte",
    )
        .unwrap();

    // Model
    let model = Sequential::from(vec![
        box Affine::new(784, 128),
        box Relu,
        box Affine::new(128, 10),
        box Relu,
    ]);

    // Optimizer
    let optimizer = optimizers::Sgd::new(0.0001);

    // Initialize model weights and optimizer params
    model.init();
    optimizer.init();

    println!("MNIST training started!");

    for (images, labels) in mnist.iter().batch(32, Mnist::collate) {
        let input = Var::with_data(images);
        let labels = Var::with_data(labels);

        let logits = model.forward(&input);
        let loss = ops::softmax_cross_entropy(&logits, &labels);

        let params = model.params().unwrap();
        let grads = diff(&loss, &params);

        println!("loss: {}", loss.data().mean());

        optimizer.update(grads);
    }

    // * Inline gradient update *
    // input.set_data();
    // label.set_data();
    // op::recompute(&grads); // super-fast
    // optimizer.update(&grads);

    // * Evaluating higher-order derivatives *

    println!("Hello, world!");
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;


    #[test]
    fn test_grad() {
        let a = Var::with_data(Tensor::randn([2, 3]));
        let b = Var::with_data(Tensor::randn([4, 2, 3]));

        let c = &a + &b;

        let grads = diff(&c, &[&a, &b]);

        let grad_a = grads.get(&a).unwrap();
        let grad_b = grads.get(&b).unwrap();

        println!("{:?}", c.data());
        println!("{:?}", grad_a.data());

        assert_eq!(grad_a.shape(), a.shape());
        assert_eq!(grad_b.shape(), b.shape());
    }
}