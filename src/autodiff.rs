use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};

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

    // The 'genesis' gy/gy (which is always 1)
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

                // gradients are accumulated
                Some(p_gx) => d_map.insert(x, p_gx + gx),
            };
        }
    }

    // aggregate outputs
    d_map.retain(|v, _| xs.contains(v));

    // map to vector
    d_map.into_iter().map(|(_, gx)| gx).collect()
}
pub struct OpNode {
    op: Box<dyn Op>,

    // Generation rank, required for topological ordering of computational nodes
    rank: u32,

    input: Vec<Rc<RefCell<VarNode>>>,
    output: Weak<RefCell<VarNode>>,
}

#[derive(Default)]
pub struct VarNode {
    data: Option<f32>,
    parent: Option<Rc<OpNode>>,
}

impl VarNode {
    fn rank(&self) -> u32 {
        match &self.parent {
            None => 0,
            Some(n) => n.rank,
        }
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
