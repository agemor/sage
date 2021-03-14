use crate::autodiff::ops::{Operation, OperationEnum, Operator};
use crate::autodiff::session::Session;
use crate::autodiff::Ranked;
use crate::tensor::shape::{Shape, ToIndex, ToShape};
use crate::tensor::Tensor;
use std::cell::{Ref, RefCell, RefMut};
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};
use std::time::Duration;

pub struct VarNode {
    pub data: Option<Tensor>,
    pub shape: Shape,
    pub origin: Option<OperationEnum>,

    // memory/time profile
    pub runtime: Option<RuntimeProfile>,
}

impl VarNode {
    pub(crate) fn order(&self) -> usize {
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
                shape: s.to_shape(0),
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

    pub(crate) fn to_weak(&self) -> WeakVar {
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

    pub fn is_leaf(&self) -> bool {
        let node = self.node();
        node.origin.is_none()
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
}

pub trait ToVar {
    fn to_var(&self) -> Var;
}

impl ToVar for Var {
    fn to_var(&self) -> Var {
        self.clone()
    }
}

impl ToVar for &Var {
    fn to_var(&self) -> Var {
        (*self).clone()
    }
}

impl ToVar for Tensor {
    fn to_var(&self) -> Var {
        Var::with_data(self.clone())
    }
}

impl ToVar for &Tensor {
    fn to_var(&self) -> Var {
        Var::with_data((*self).clone())
    }
}

impl ToVar for f32 {
    fn to_var(&self) -> Var {
        Var::with_data(Tensor::scalar(*self))
    }
}

impl ToVar for isize {
    fn to_var(&self) -> Var {
        Var::with_data(Tensor::scalar(*self as f32))
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

pub struct RuntimeProfile {
    pub mem_store: usize,

    pub call_time: Duration,
}

pub struct WeakVar {
    node: Weak<RefCell<VarNode>>,
}

impl WeakVar {
    fn from(var: &Var) -> Self {
        WeakVar {
            node: Rc::downgrade(&var.node),
        }
    }

    pub(crate) fn to_var(&self) -> Option<Var> {
        self.node.upgrade().map(|x| Var::from_node(x))
    }
}
