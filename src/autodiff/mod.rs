use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use itertools::Itertools;

use session::Session;

use crate::autodiff::var::Var;
use crate::tensor::Tensor;
use std::ops::Add;
use std::time::Instant;

pub mod ops;
pub mod session;
pub mod sim;
pub mod var;

// Differentiate variables, a.k.a. backpropagation
pub fn diff(y: &Var, xs: &[&Var]) -> HashMap<Var, Var> {
    let start_time = Instant::now();

    let mut queue = BinaryHeap::<Ranked<Var>>::new();
    let mut grads = HashMap::<Var, Var>::new();

    // The 'genesis' gy/gy, (which always equals to 1)
    grads.insert(y.clone(), Var::with_data(Tensor::ones(y.shape())));
    queue.push(y.clone().into_ranked());

    let mut iterations = 0;

    while !queue.is_empty() {
        iterations += 1;

        if iterations % 1000 == 0 {
            println!(
                "[{}] queue size: {}, elapsed time: {} sec",
                iterations,
                queue.len(),
                start_time.elapsed().as_millis() as f32 / 1000.0
            );
        }

        // must unwrap
        let y = queue.pop().unwrap().into_inner();
        let gy = grads.get(&y).unwrap();

        let y_node = y.node();

        if let Some(ref operation) = y_node.origin {
            let x = operation.input();
            let gx = operation.input_adjoint(gy);

            // insert (x, gx) pairs into grads hashmap
            for (x, gx) in x.into_iter().zip(gx.iter()) {
                if !grads.contains_key(&x) {
                    queue.push(x.clone().into_ranked())
                }
                grads
                    .entry(x.clone())
                    .and_modify(|v| *v = gx.add(&*v))
                    .or_insert_with(|| gx.clone());
            }
        }
    }

    println!("[{}] backward pass generated. ", { iterations });

    // aggregate outputs... unused gradients are dropped.
    grads.retain(|ref v, _| xs.contains(v));
    grads
}

pub fn diff_eval(y: &Var, xs: &[&Var]) -> HashMap<Var, Var> {
    let grads = diff(y, xs);

    let grad_vec = grads.values().cloned().collect_vec();

    // evaluate grads
    Session::with_budget(grad_vec, 10000).eval();

    grads
}

pub struct Ranked<T> {
    inner: T,
    rank: usize,
}

impl<T> Ranked<T> {
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> Eq for Ranked<T> {}

impl<T> PartialEq for Ranked<T> {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<T> Ord for Ranked<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

impl<T> PartialOrd for Ranked<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
