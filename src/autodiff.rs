use std::cell::{Ref, RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{Hash, Hasher};
use std::mem;
use std::rc::{Rc, Weak};

use std::time::Duration;

use crate::ops;
use crate::ops::{matmul, reshape, transpose};
use crate::session::Session;
use crate::shape::{Shape, ToIndex, ToShape};
use crate::tensor::Tensor;
use itertools::Itertools;
use std::ops::Deref;

pub trait Operator<const N: usize> {
    fn compute(&self, x: [&Tensor; N]) -> Tensor;

    fn forward(self, x: [&Var; N]) -> Var;

    // f(x) = u
    // f'(x) -> ∂u/∂x
    // f'(x) * gy = ∂u/∂x * ∂y/∂u = ∂y/∂x
    fn backward(&self, x: [&Var; N], gy: &Var) -> [Var; N];

    fn mem_req(&self) -> usize {
        mem::size_of::<f32>()
    }
}

pub struct Operation<const N: usize> {
    operator: Box<dyn Operator<{ N }>>,
    order: usize,

    input: [Var; N],
    output: WeakVar, // to prevent cyclic references
}

impl<const N: usize> Operation<{ N }> {
    pub fn new<O>(operator: O, input: [Var; N], output: Var) -> Self
    where
        O: Operator<{ N }> + 'static,
    {
        let order = input
            .iter()
            .map(|a| a.node().order()) // each rank
            .max() // get max rank in parent gen
            .unwrap()
            + 1; // increase + 1

        Operation {
            operator: Box::new(operator),
            order,
            input,
            output: output.to_weak(),
        }
    }

    pub fn compute(&self) -> Tensor {
        let data = self.input.each_ref().map(|v| v.data());
        let data_borrowed = data.each_ref().map(|t| t.deref());
        self.operator.compute(data_borrowed)
    }

    pub fn mem_req(&self) -> usize {
        self.operator.mem_req()
    }

    pub fn input(&self) -> [Var; N] {
        self.input.clone()
    }

    pub fn output(&self) -> Var {
        self.output.to_var().unwrap()
    }

    pub fn input_adjoint(&self, output_adjoint: &Var) -> [Var; N] {
        let input = self.input.each_ref();
        self.operator.backward(input, output_adjoint)
    }
}

pub enum OperationEnum {
    Unary(Operation<1>),
    Binary(Operation<2>),
}

// TODO: write some macros
impl OperationEnum {
    pub fn arity(&self) -> usize {
        match self {
            Self::Unary(_) => 1,
            Self::Binary(_) => 2,
        }
    }

    pub fn order(&self) -> usize {
        match self {
            Self::Unary(o) => o.order,
            Self::Binary(o) => o.order,
        }
    }

    pub fn compute(&self) -> Tensor {
        match self {
            Self::Unary(o) => o.compute(),
            Self::Binary(o) => o.compute(),
        }
    }

    pub fn mem_req(&self) -> usize {
        match self {
            Self::Unary(o) => o.mem_req(),
            Self::Binary(o) => o.mem_req(),
        }
    }

    pub fn input(&self) -> Vec<Var> {
        // TODO: change it to smallvec for better performance
        match self {
            Self::Unary(o) => o.input.to_vec(),
            Self::Binary(o) => o.input.to_vec(),
        }
    }

    pub fn output(&self) -> Option<Var> {
        match self {
            Self::Unary(o) => o.output.to_var(),
            Self::Binary(o) => o.output.to_var(),
        }
    }

    pub fn input_adjoint(&self, output_adjoint: &Var) -> Vec<Var> {
        match self {
            Self::Unary(o) => o.input_adjoint(output_adjoint).to_vec(),
            Self::Binary(o) => o.input_adjoint(output_adjoint).to_vec(),
        }
    }
}

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

pub struct RuntimeProfile {
    pub mem_store: usize,

    pub call_time: Duration,
}

pub struct VarNode {
    pub data: Option<Tensor>,
    pub shape: Shape,
    pub origin: Option<OperationEnum>,

    // memory/time profile
    pub runtime: Option<RuntimeProfile>,
}

impl VarNode {
    fn order(&self) -> usize {
        match &self.origin {
            None => 0,
            Some(o) => o.order(),
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
        if let Some(ref operation) = self.origin {
            let mut time = Duration::new(0, 0);
            let mut stack = Vec::<Var>::new();

            // must unwrap. "I, myself is the proof"
            stack.push(operation.output().unwrap());

            // start with self..
            while !stack.is_empty() {
                let var = stack.pop().unwrap();
                let var_node = var.node();

                // overhead of op_node itself
                // must unwrap, as un-evaluated VarNodes are not in the actives set
                let runtime = var_node.runtime.as_ref().unwrap();

                time += runtime.call_time;

                if let Some(ref operation) = var_node.origin {
                    for v in operation.input() {
                        if !v.is_evaluated() {
                            stack.push(v)
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
    /////// Var constructors ///////

    pub fn with_shape<S>(s: S) -> Var
    where
        S: ToShape,
    {
        Var {
            node: Rc::new(RefCell::new(VarNode {
                data: None,
                shape: s.to_shape(),
                origin: None,
                runtime: Some(RuntimeProfile {
                    mem_store: 0,
                    call_time: Duration::ZERO,
                }),
            })),
        }
    }

    pub fn with_data(data: Tensor) -> Var {
        Var {
            node: Rc::new(RefCell::new(VarNode {
                shape: data.shape(),
                origin: None,
                runtime: Some(RuntimeProfile {
                    mem_store: data.mem_size(),
                    call_time: Duration::ZERO,
                }),
                data: Some(data),
            })),
        }
    }

    fn from_node(node: Rc<RefCell<VarNode>>) -> Var {
        Var { node }
    }

    /////// Var constructors (from operation) ///////

    pub fn from_unary_op<S, O>(shape: S, operator: O, arg: &Var) -> Self
    where
        S: ToShape,
        O: Operator<1> + 'static,
    {
        let var = Var::with_shape(shape);

        let input = [arg.clone()];
        let output = var.clone();

        let operation = Operation::new(operator, input, output);
        var.node_mut().origin = Some(OperationEnum::Unary(operation));
        var
    }

    pub fn from_binary_op<S, O>(shape: S, operator: O, args: [&Var; 2]) -> Self
    where
        S: ToShape,
        O: Operator<2> + 'static,
    {
        let var = Var::with_shape(shape);

        let input = [args[0].clone(), args[1].clone()];
        let output = var.clone();

        let operation = Operation::new(operator, input, output);

        var.node_mut().origin = Some(OperationEnum::Binary(operation));
        var
    }

    /////// Var converters ///////

    pub fn into_ranked(self) -> Ranked<Self> {
        let rank = self.node().order();

        Ranked { inner: self, rank }
    }

    fn to_weak(&self) -> WeakVar {
        WeakVar::from(self)
    }

    /////// Node accesses ///////

    pub fn node(&self) -> Ref<VarNode> {
        RefCell::borrow(&self.node)
    }

    pub fn node_mut(&self) -> RefMut<VarNode> {
        RefCell::borrow_mut(&self.node)
    }

    pub fn is_evaluated(&self) -> bool {
        let node = self.node();
        node.data.is_some()
    }

    fn data_unchecked(&self) -> Ref<Tensor> {
        let node = self.node();
        Ref::map(node, |x| match x.data {
            None => panic!("unevaluated data!"),
            Some(ref u) => u,
        })
    }

    // retrieve tensor, evaluate if does not have one.
    pub fn data(&self) -> Ref<Tensor> {
        if !self.is_evaluated() {
            let mut session = Session::with_budget(vec![self.clone()], 10000);
            session.eval();
        }

        self.data_unchecked()
    }

    pub fn rank(&self) -> usize {
        self.node().shape.len()
    }

    pub fn shape(&self) -> Shape {
        self.node().shape
    }

    pub fn update_grads(&self, grad: Tensor) {
        let mut node = self.node_mut();
        let param = node.data.as_ref().unwrap();

        let new_param = param - grad;

        node.data = Some(new_param);
    }

    pub fn set_data(&self, data: Tensor) {
        self.node_mut().data = Some(data);
    }

    /////////// math functions+

    pub fn matmul(&self, other: &Var) -> Var {
        matmul(self, other)
    }

    pub fn transpose<I, J>(&self, axis_a: I, axis_b: J) -> Var
    where
        I: ToIndex,
        J: ToIndex,
    {
        transpose(self, axis_a, axis_b)
    }

    pub fn reshape<S>(&self, shape: S) -> Var
    where
        S: ToShape,
    {
        reshape(self, shape)
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

impl AsRef<Var> for Var {
    fn as_ref(&self) -> &Var {
        self
    }
}

struct WeakVar {
    node: Weak<RefCell<VarNode>>,
}

impl WeakVar {
    fn from(var: &Var) -> Self {
        WeakVar {
            node: Rc::downgrade(&var.node),
        }
    }

    fn to_var(&self) -> Option<Var> {
        self.node.upgrade().map(|x| Var::from_node(x))
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
