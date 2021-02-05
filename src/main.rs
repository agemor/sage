#![feature(duration_constants)]
#![feature(duration_zero)]
#[macro_use]
extern crate impl_ops;

mod autodiff;
mod net;
mod op;
mod tensor;

fn main() {
    println!("Hello, world!");
}
