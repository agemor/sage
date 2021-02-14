use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem;
use std::rc::{Rc, Weak};

use std::time::{Duration, Instant};

use crate::tensor::{Shape, ShapeError, Tensor};
use crate::{op, tensor};

pub trait Op {
    fn compute(&self, x: &[&Tensor]) -> Tensor;

    fn forward(&self, x: &[&Var]) -> Result<Shape, ShapeError> {
        x.iter().try_fold(x[0].shape().clone(), |s, x| {
            // try broadcasting
            s.broadcast(x.shape())
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

// Differentiate variables
pub fn diff<'a>(y: &'a Var, xs: &[&'a Var]) -> HashMap<&'a Var, Var> {
    let mut queue = BinaryHeap::<&OpNode>::new();
    let mut d_map = HashMap::<&Var, Var>::new();

    // The 'genesis' gy/gy, (which always equals to 1)
    d_map.insert(y, Var::from_tensor(tensor::ones(y.shape())));

    // Add first op_node to the queue!
    if let Some(ref op_node) = y.node().parent {
        queue.push(op_node);
    }

    while !queue.is_empty() {
        // must unwrap!!!
        let op_node = queue.pop().unwrap();

        let x = op_node.input_vars();
        let y = op_node.output_var().unwrap();

        let gy = d_map.get(&y).unwrap();
        let gx = op_node.op.backward(&x.iter().collect::<Vec<&Var>>(), gy);

        // dispatch each gy with its op_node
        for (x, gx) in x.iter().zip(gx) {
            match d_map.get(x) {
                None => d_map.insert(x, gx),

                // gradients are accumulated TODO: memory really freed?
                Some(gx_prev) => d_map.insert(x, op::add(gx_prev, &gx)),
            };
        }
    }

    // aggregate outputs.. magic happens here
    d_map.retain(|v, _| xs.contains(v));
    d_map
}

// Evaluate variables
pub fn eval(xs: &[&Var]) {
    // available memory at this moment
    let mut mem_budget = 10000;

    let mut actives = HashSet::<Var>::new();

    let mut stack: Vec<Var> = Vec::new();

    // variables that have any dependencies... ok to drop now.
    // collecting "garbage"
    let clear_indep = || {
        // build deps set
        let mut deps: HashSet<Var> = HashSet::new();

        // get last unevaluated leaves

        for x in xs.iter() {
            let mut n_stack: Vec<Var> = vec![*x.clone()];

            while !n_stack.is_empty() {
                let var = n_stack.pop().unwrap();

                deps.insert(var);

                // we don't care about the leaf nodes, since they are already stuffed with data.
                if let Some(ref parent) = var.node().parent {
                    // if evaluated, insert.
                    // if not evaluated add to the deps and n_stack.
                    for in_var in parent.input_vars().iter() {
                        // this saves some redundant operations
                        if !deps.contains(in_var) {
                            if let None = in_var.data() {
                                n_stack.push(in_var.clone());
                            }
                        }
                    }
                }
            }
        }

        //
        for var in actives.iter() {
            if !deps.contains(var) {
                // free acts
                var.node().free_data();
                actives.remove(var);
            }
        }
    };

    //

    let greedy_drop = |must_keep: &[Var], mem_req: usize| {
        // free ones that have minimum T/M

        // do not drop current dependencies & leaf

        let mut mem_freed = 0;

        while mem_freed < mem_req {
            let m = actives
                .iter()
                .filter(|v| !must_keep.contains(v)) // do not touch must_keep
                .min_by_key(|v| {
                    let n = v.node();

                    let mem_cost = n.runtime.unwrap().mem_store as f64;
                    let time_cost = n.recompute_cost().as_secs_f64();

                    time_cost / mem_cost;
                });

            if let Some(v) = m {
                v.node().free_data();

                mem_freed += v.node().runtime.unwrap().mem_store;
                actives.remove(v);
            } else {
                panic!("cannot free more!!");
            }
        }
    };

    while !stack.is_empty() {
        let var = { stack.last().unwrap() };

        let mut node = var.node_mut();

        // if not evaluated...
        if let None = node.data {
            // it is very awkward to not have parents,,... with no data
            if let Some(ref parent) = node.parent {
                // if all evaluated... eval self and done.

                let ready = parent
                    .input_vars()
                    .into_iter()
                    .fold(true, |ready, x| match x.data() {
                        None => {
                            stack.push(x);
                            false
                        }
                        Some(_) => ready,
                    });

                if ready {
                    //err
                    stack.pop();

                    let in_vars = parent.input_vars();

                    // exceeds mem budget?
                    if parent.op.mem_req() > mem_budget {
                        // clear out zero deps.
                        clear_indep();

                        // only when necessary
                        greedy_drop(&in_vars, parent.op.mem_req());
                    }

                    let in_tensors = in_vars
                        .into_iter()
                        .map(|v| {
                            // must unwrap (checked in 'ready' phase)
                            v.data().unwrap()
                        })
                        .collect::<Vec<&Tensor>>();

                    // do some runtime profiling
                    let timer = Instant::now();

                    let out_tensor = parent.op.compute(&in_tensors);

                    let profile = RuntimeProfile {
                        mem_store: tensor::mem_size(&out_tensor),
                        call_time: timer.elapsed(),
                    };

                    node.data = Some(out_tensor);
                    node.runtime = Some(profile);

                    // register actives
                    actives.insert(var.clone());
                }
            }
            // null leaf
            else {
                panic!("null leaf");
            }
        }
    }
}

pub struct OpNode {
    op: Box<dyn Op>,

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

    fn input_vars(&self) -> Vec<Var> {
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

impl Eq for OpNode {}

impl PartialEq for OpNode {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl Ord for OpNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

impl PartialOrd for OpNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct RuntimeProfile {
    mem_store: usize,

    call_time: Duration,
}

pub struct VarNode {
    data: Option<Tensor>,
    shape: Shape,
    parent: Option<OpNode>,

    // memory/time profile
    runtime: Option<RuntimeProfile>,
}

impl VarNode {
    fn rank(&self) -> usize {
        match &self.parent {
            None => 0,
            Some(n) => n.rank,
        }
    }

    // get re-computation cost, in a dynamic-programming fashion.
    fn recompute_cost(&self) -> Duration {
        if let Some(ref n) = self.parent {
            let mut stack = vec![n];

            // must unwrap, as un-evaluated VarNodes are not in the actives set
            let mut time = Duration::new(0, 0);

            // start with self..
            while !stack.is_empty() {
                let op_node = stack.pop().unwrap();

                // overhead of op_node itself
                let runtime = op_node
                    .output_var()
                    .unwrap() // must unwrap, as only op_nodes with outputs are in the stack
                    .node()
                    .runtime
                    .unwrap(); // must unwrap, as un-evaluated VarNodes are not in the actives set

                time += runtime.call_time;

                for in_var in op_node.input_vars().iter() {
                    // needs re-computation
                    if let None = in_var.data() {
                        if let Some(ref parent) = in_var.node().parent {
                            stack.push(parent)
                        }
                        // no parent and no data?? wtf?
                        else {
                            panic!("something gone wrong");
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

    fn free_data(&mut self) {
        self.data = None;
    }
}

pub struct Var {
    node: Rc<RefCell<VarNode>>,
}

impl Var {
    fn with_shape(shape: Shape) -> Var {
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
                data: Some(data),
                shape: Shape::new(data.shape()),
                parent: None,
                runtime: Some(RuntimeProfile {
                    mem_store: tensor::mem_size(&data),
                    call_time: Duration::ZERO,
                }),
            })),
        }
    }

    fn node(&self) -> &VarNode {
        &RefCell::borrow(&self.node)
    }

    fn node_mut(&self) -> &mut VarNode {
        &mut RefCell::borrow_mut(&self.node)
    }

    fn from_node(node: Rc<RefCell<VarNode>>) -> Var {
        Var { node }
    }

    fn data(&self) -> Option<&Tensor> {
        self.node().data.map_or(None, |e| Some(&e))
    }

    pub fn shape(&self) -> &Shape {
        &self.node().shape
    }

    fn set_data(&self) {}
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
