use rand_distr::{Normal, Uniform};

use crate::tensor::shape::ToShape;
use crate::tensor::Tensor;

pub fn kaiming_uniform<S>(shape: S, gain: f32) -> Tensor
where
    S: ToShape,
{
    let shape = shape.to_shape();

    let (fan_in, _) = fan_in_and_out(&shape);
    let std = gain * (1.0 / fan_in as f32).sqrt();
    let a = 3.0_f32.sqrt() * std;

    Tensor::from_dist(shape, Uniform::new(-a, a))
}

pub fn kaiming_normal<S>(shape: S, gain: f32) -> Tensor
where
    S: ToShape,
{
    let shape = shape.to_shape();

    let (fan_in, _) = fan_in_and_out(&shape);
    let std = gain * (1.0 / fan_in as f32).sqrt();

    Tensor::from_dist(shape, Normal::new(0.0, std).unwrap())
}

fn fan_in_and_out<S>(shape: S) -> (usize, usize)
where
    S: ToShape,
{
    let shape = shape.to_shape();

    if shape.len() < 2 {
        panic!("cannot compute.. shape too small");
    }

    let num_in_fmaps = shape[1];
    let num_out_fmaps = shape[0];

    let mut receptive_field_size = 1;

    if shape.len() > 2 {
        receptive_field_size = shape.sizes()[2..].iter().fold(1, |a, b| a * (*b));
    }

    let fan_in = num_in_fmaps * receptive_field_size;
    let fan_out = num_out_fmaps * receptive_field_size;

    (fan_in, fan_out)
}
