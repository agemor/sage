use std::cell::{Ref, RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{Hash, Hasher};
use std::mem;
use std::rc::{Rc, Weak};

use std::time::Duration;

use crate::session::Session;
use crate::tensor::{Shape, ShapeError, Tensor};
use crate::{op, tensor};

pub trait Op {
    fn compute(&self, x: &[&Tensor]) -> Tensor;

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        x.iter().try_fold(x[0].shape().clone(), |s, x| {
            // try broadcasting
            s.broadcast(&x.shape())
        })
    }

    fn mem_req(&self) -> usize {
        mem::size_of::<f32>()
    }

    // f(x) = u
    // f'(x) -> ∂u/∂x
    // f'(x) * gy = ∂u/∂x * ∂y/∂u = ∂y/∂x
    fn backward(&self, x: &[&Var], gy: &Var) -> Vec<Var>;
}

// Create a tensor operation
pub fn op(op: Box<dyn Op>, args: &[&Var]) -> Var {
    match op.forward(args) {
        Ok(shape) => {
            let out = Var::with_shape(shape.clone());
            let op_node = OpNode::new(op, args, &out);

            out.node_mut().parent = Some(op_node);
            out
        }
        Err(err) => {
            panic!(err.msg)
        }
    }
}

// Differentiate variables, a.k.a. backpropagation
pub fn diff(y: &Var, xs: &[&Var]) -> HashMap<Var, Var> {
    let mut queue = BinaryHeap::<Ranked<Var>>::new();
    let mut grads = HashMap::<Var, Var>::new();

    // The 'genesis' gy/gy, (which always equals to 1)
    grads.insert(y.clone(), Var::from_tensor(tensor::ones(&y.shape())));

    queue.push(y.clone().into_ranked());

    while !queue.is_empty() {
        // must unwrap
        let var = queue.pop().unwrap().into_inner();
        let var_node = var.node();

        if let Some(ref parent) = var_node.parent {
            let y = parent.output_var().unwrap(); // == var.clone()
            let x = parent.input_vars();

            let gy = grads.get(&y).unwrap(); // must unwrap
            let gx = parent.op.backward(&x.iter().collect::<Vec<&Var>>(), gy);

            // dispatch each x
            for (x, gx) in x.into_iter().zip(gx) {
                // update gradients
                grads
                    .entry(x.clone())
                    .and_modify(|v| *v = op::add(v, &gx))
                    .or_insert(gx); // new

                queue.push(x.into_ranked())
            }
        }
    }

    // aggregate outputs... unused gradients are dropped.
    grads.retain(|ref v, _| xs.contains(v));
    grads
}

pub struct OpNode {
    pub(crate) op: Box<dyn Op>,

    // Generation rank, required for topological ordering of computational nodes
    rank: usize,

    input: Vec<Rc<RefCell<VarNode>>>,
    output: Weak<RefCell<VarNode>>,
}

impl OpNode {
    fn new(op: Box<dyn Op>, input: &[&Var], output: &Var) -> Self {
        let rank = input
            .into_iter()
            .map(|a| a.node().rank()) // each rank
            .max() // get max rank in parent gen
            .unwrap()
            + 1; // increase + 1

        OpNode {
            op,
            rank,
            input: input.iter().map(|e| e.node.clone()).collect(),
            output: Rc::downgrade(&output.node),
        }
    }

    pub(crate) fn input_vars(&self) -> Vec<Var> {
        self.input
            .iter()
            .map(|e| Var::from_node(e.clone()))
            .collect()
    }

    fn output_var(&self) -> Option<Var> {
        self.output
            .upgrade() // try upgrade
            .map_or(None, |e| Some(Var::from_node(e.clone())))
    }
}

pub struct RuntimeProfile {
    pub mem_store: usize,

    pub call_time: Duration,
}

pub struct VarNode {
    pub data: Option<Tensor>,
    pub shape: Shape,
    pub parent: Option<OpNode>,

    // memory/time profile
    pub runtime: Option<RuntimeProfile>,
}

impl VarNode {
    fn rank(&self) -> usize {
        match &self.parent {
            None => 0,
            Some(n) => n.rank,
        }
    }

    pub fn recompute_heuristic(&self) -> Option<u128> {
        if let Some(ref runtime) = self.runtime {
            let time = self.recompute_time().as_nanos();
            let space = runtime.mem_store as u128;
            Some(time / space)
        } else {
            None
        }
    }

    // get re-computation cost, in a dynamic-programming fashion.
    fn recompute_time(&self) -> Duration {
        if let Some(ref parent) = self.parent {
            let mut time = Duration::new(0, 0);
            let mut stack = Vec::<Var>::new();

            if let Some(var) = parent.output_var() {
                stack.push(var);
            }

            // start with self..
            while !stack.is_empty() {
                let var = stack.pop().unwrap();
                let var_node = var.node();

                // overhead of op_node itself
                // must unwrap, as un-evaluated VarNodes are not in the actives set
                let runtime = var_node.runtime.as_ref().unwrap();

                time += runtime.call_time;

                if let Some(ref parent) = var_node.parent {
                    let in_vars = parent.input_vars();

                    for in_var in in_vars.iter() {
                        // needs re-computation

                        if !in_var.is_evaluated() {
                            stack.push(in_var.clone())
                        }
                    }
                }
            }
            time
        } else {
            // no parent -> cannot be recomputed
            Duration::MAX
        }
    }

    pub(crate) fn free_data(&mut self) {
        self.data = None;
    }
}

pub struct Var {
    node: Rc<RefCell<VarNode>>,
}

impl Var {
    pub fn with_shape(shape: Shape) -> Var {
        Var {
            node: Rc::new(RefCell::new(VarNode {
                data: None,
                shape,
                parent: None,
                runtime: Some(RuntimeProfile {
                    mem_store: 0,
                    call_time: Duration::ZERO,
                }),
            })),
        }
    }

    pub fn from_tensor(data: Tensor) -> Var {
        Var {
            node: Rc::new(RefCell::new(VarNode {
                shape: Shape::new(&data.shape()),
                parent: None,
                runtime: Some(RuntimeProfile {
                    mem_store: tensor::mem_size(&data),
                    call_time: Duration::ZERO,
                }),
                data: Some(data),
            })),
        }
    }

    pub(crate) fn into_ranked(self) -> Ranked<Self> {
        let rank = self.node().rank();

        Ranked { inner: self, rank }
    }

    pub(crate) fn node(&self) -> Ref<VarNode> {
        RefCell::borrow(&self.node)
    }

    pub(crate) fn node_mut(&self) -> RefMut<VarNode> {
        RefCell::borrow_mut(&self.node)
    }

    fn from_node(node: Rc<RefCell<VarNode>>) -> Var {
        Var { node }
    }

    pub fn is_evaluated(&self) -> bool {
        let node = self.node();
        node.data.is_some()
    }

    pub fn data_unchecked(&self) -> Ref<Tensor> {
        let node = self.node();
        Ref::map(node, |x| match x.data {
            None => panic!("unevaluated data!"),
            Some(ref u) => u,
        })
    }

    // retrieve tensor, evaluate if does not have one.
    pub fn data(&self) -> Ref<Tensor> {
        if self.is_evaluated() {
            let mut session = Session::with_budget(vec![self.clone()], 0);
            session.eval();
        }

        self.data_unchecked()
    }

    pub fn shape(&self) -> Shape {
        self.node.borrow().shape.clone()
    }

    pub fn set_data(&self, data: Tensor) {
        self.node_mut().data = Some(data);
    }
}

impl Eq for Var {}
impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.node, &other.node)
    }
}

impl Clone for Var {
    fn clone(&self) -> Self {
        Var {
            node: self.node.clone(),
        }
    }
}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.as_ptr().hash(state)
    }
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
