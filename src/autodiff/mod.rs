use std::cell::{Ref, RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::time::Duration;

use itertools::Itertools;

use session::Session;

use crate::autodiff::var::Var;
use crate::tensor::shape::{Shape, ToIndex, ToShape};
use crate::tensor::Tensor;

pub mod ops;
pub mod session;
pub mod sim;
pub mod var;

// Differentiate variables, a.k.a. backpropagation
pub fn diff(y: &Var, xs: &[&Var]) -> HashMap<Var, Var> {
    let mut queue = BinaryHeap::<Ranked<Var>>::new();
    let mut grads = HashMap::<Var, Var>::new();

    // The 'genesis' gy/gy, (which always equals to 1)
    grads.insert(y.clone(), Var::with_data(Tensor::ones(y.shape())));
    queue.push(y.clone().into_ranked());

    while !queue.is_empty() {
        // must unwrap
        let y = queue.pop().unwrap().into_inner();
        let gy = grads.get(&y).unwrap();

        let y_node = y.node();

        if let Some(ref operation) = y_node.origin {
            let x = operation.input();
            let gx = operation.input_adjoint(gy);

            // insert (x, gx) pairs into grads hashmap
            for (x, gx) in x.into_iter().zip(gx.iter()) {
                grads
                    .entry(x.clone())
                    .and_modify(|v| *v = ops::add(v, gx))
                    .or_insert_with(|| gx.clone());

                queue.push(x.into_ranked())
            }
        }
    }

    // aggregate outputs... unused gradients are dropped.
    grads.retain(|ref v, _| xs.contains(v));

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
