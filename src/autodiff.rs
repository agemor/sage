use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, BTreeSet};
use std::hash::{Hash, Hasher};
use std::ops;
use std::rc::{Rc, Weak};

use crate::op;

pub trait Op {
    fn compute(&self, x: &[f32]) -> f32;
    // f(x) = u
    // f'(x) -> ∂u/∂x
    // f'(x) * gy = ∂u/∂x * ∂y/∂u = ∂y/∂x
    fn diff(&self, x: &[&Var], gy: &Var) -> Vec<Var>;
}

pub fn op(op: Box<dyn Op>, x: &[&Var]) -> Var {
    let in_nodes = x.iter().map(|x| x.node).collect();
    let in_ranks: Vec<u32> = x.iter().map(|x| x.node.as_ref().borrow().rank()).collect();

    let out_var_node = Rc::new(RefCell::new(VarNode {
        data: None,   // Not evaluated yet..!
        parent: None, // will be filled later
    }));

    let node = Rc::new(OpNode {
        op,
        rank: in_ranks.iter().max().unwrap() + 1,
        input: in_nodes,
        output: Rc::downgrade(&out_var_node),
    });

    out_var_node.borrow_mut().parent = Some(node.clone());

    Var::from_node(out_var_node)
}

pub fn diff(y: &Var, xs: &[&Var]) -> Vec<Var> {
    let mut queue = BinaryHeap::new();
    let mut d_map: HashMap<&Var, Var> = HashMap::new();

    // The 'genesis' gy/gy
    d_map.insert(y, Var::new(1.0));

    // Add to the queue!
    if let Some(n) = &y.node.as_ref().borrow().parent {
        queue.push(n.clone());
    }

    while !queue.is_empty() {
        // must unwrap!!!
        let op_node = queue.pop().unwrap();

        let out_var = Var::from_node(op_node.output.upgrade().unwrap());
        let in_vars: Vec<Var> = op_node
            .input
            .iter()
            .map(|vn| Var::from_node(vn.clone()))
            .collect();

        let in_var_ref = in_vars.iter().collect::<Vec<&Var>>();
        let gy = d_map.get(&out_var).unwrap();
        let new_gy = op_node.op.diff(&in_var_ref, gy);

        // dispatch each gy with its op_node
        for (x, gx) in in_vars.iter().zip(new_gy) {
            match d_map.get(x) {
                None => d_map.insert(x, gx),

                // gradients are accumulated TODO: memory really freed?
                Some(p_gx) => d_map.insert(x, p_gx + gx),
            };
        }
    }

    // aggregate outputs.. magic happens here
    d_map.retain(|v, _| xs.contains(v));

    // map to vector TODO: make same ordering!
    d_map.into_iter().map(move |(_, gx)| gx).collect()
}

pub fn eval(xs: &[&Var]) {
    // dependency counter

    let mut actives: BTreeSet<Var> = BTreeSet::new();

    // fill actives

    let mut stack: Vec<Var> = Vec::new();

    // increase counter of each x

    // variables that have any dependencies... ok to drop now.
    let zero_dep_clear = |x| {
        // build deps set
        let mut deps: HashSet<Var> = HashSet::new();

        // get last unevaluated leaves

        for x in xs.iter() {
            let mut n_stack: Vec<Var> = vec![x.clone()];

            while !n_stack.is_empty() {
                let var = n_stack.pop().unwrap();
                let node = var.node.borrow();

                deps.insert(var);

                // we don't care about the leaf nodes, since they are already stuffed with data.
                if let Some(ref parent) = node.parent {
                    // if evaluated, insert.
                    // if not evaluated add to the deps and n_stack.
                    for in_varn in parent.input.iter() {
                        let in_var = Var::from_node(in_varn.clone());

                        // this saves some redundant operations
                        if !deps.contains(&in_var) {
                            if let None = in_varn.borrow().data {
                                n_stack.push(Var::from_node(in_varn.clone()));
                            }
                        }
                    }
                }
            }
        }

        //
        for acts in actives.iter() {

            if !deps.contains(acts) {
                // free acts
                acts.node.borrow().free_data();

                actives.remove(acts);
            }

            // do this until memory budget is met
        }

    };

    //
    let greedy_drop = |x| {

        for acts in actives.iter() {

            // free ones that have minimum T/M


        }

    };

    // DPS
    let mut oom = false;

    while !stack.is_empty() {
        let var = stack.last().unwrap();

        let mut node = var.node.borrow_mut();

        // if not evaluated...
        if let None = node.data {
            // it is very awkward to not have parents,,... with no data
            if let Some(ref parent) = node.parent {
                // if all evaluated... eval self and done.

                let ready = parent
                    .input
                    .iter()
                    .fold(true, |acc, x| match x.borrow().data {
                        None => {
                            stack.push(Var::from_node(x.clone()));
                            false
                        }
                        Some(_) => acc,
                    });

                if ready {
                    //err
                    stack.pop();

                    let in_args = parent
                        .input
                        .iter()
                        .map(|x| x.borrow().data.unwrap())
                        .collect();

                    let out = parent.op.compute(&in_args);
                    node.data = Some(out);

                    // register actives
                    actives.insert(var.clone());

                    // if out of memory,
                    if oom {}
                }
            }
            // null leaf
            else {
                panic!("null leaf");
            }
        }
    }

    // memory budget M

    // drop activations when exceeds memory budget

    // first -> no dependencies (dropped when consumed)
    // second -> \s

    // M = model memory, optimizer memory, checkpoints (+outputs), peak buffer memory

    // schedule checkpoints less than memory budgets..

    // greedy activation drop

    // greedy selection of node with max time_cost() / mem_cost()

    // greedy tree evaluation

    // calculate modules with larger memory requirements
    //
    //  children = [nodes...]
    //  eval each children with less cp size
    //
}

//
// self memory hold + peak memory usage of children
//
fn mem_cost() {}

fn time_cost() {}

pub struct OpNode {
    op: Box<dyn Op>,

    // Generation rank, required for topological ordering of computational nodes
    rank: u32,

    input: Vec<Rc<RefCell<VarNode>>>,
    output: Weak<RefCell<VarNode>>,
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

#[derive(Default)]
pub struct VarNode {
    data: Option<f32>,

    parent: Option<OpNode>,
}

impl VarNode {
    fn rank(&self) -> u32 {
        match &self.parent {
            None => 0,
            Some(n) => n.rank,
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
    pub fn new(data: f32) -> Var {
        Var {
            node: Rc::new(RefCell::new(VarNode {
                data: Some(data),
                parent: None,
            })),
        }
    }

    fn from_node(node: Rc<RefCell<VarNode>>) -> Var {
        Var { node }
    }

    fn data(&self) -> f32 {
        0.0
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

// bunch of operator loadings...

impl ops::Add<&Var> for &Var {
    type Output = Var;

    fn add(self, rhs: &Var) -> Var {
        op::add(self, rhs)
    }
}

impl ops::Add<&Var> for &Var {
    type Output = Var;

    fn add(self, rhs: &Var) -> Var {
        op::add(self, rhs)
    }
}
