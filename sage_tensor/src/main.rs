#![feature(is_sorted)]
#![feature(once_cell)]

use crate::backend::Backend;
use crate::tensor::Tensor;

#[macro_use]
extern crate vulkano;
extern crate vulkano_shaders;

#[macro_use]
pub mod backend;
pub mod iter;

pub mod tensor;
#[macro_use]
pub mod ops;
#[macro_use]
mod shape;
mod format;

fn main() {
    println!("GPU enabled: {}", Backend::Vulkan(0).is_available());

    let a = Tensor::from_elem([5, 7], 1.0, Backend::Vulkan(0));
    let b = Tensor::from_elem([5, 7], 2.0, Backend::Vulkan(0));

    let c = Tensor::concat(&[&a, &b], 1);
    println!("{:?}", c);
    //
    // let b = Tensor::<f32>::randn([4, 3, 5, 7], a.backend());
    //
    // println!("{:?}", &b);

    //
    // println!("calc ok!");
    // println!("{:?}", c);
    //
    // c.iter().for_each(|(i, v)| {
    //     println!("{:?}", (i, v));
    // });
}
