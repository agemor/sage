// computational costs of each function
// profiled manually

// backward graph

use crate::autodiff::var::RuntimeProfile;
use crate::autodiff::Var;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

// Variable evaluation session
pub struct Sim {
    pub mem_budget: usize,
    pub mem_used: usize,
    pub peak_mem_used: usize,

    pub elapsed_time: Duration,

    pub targets: Vec<Var>,

    pub resolved: HashSet<Var>,
}

///
/// benchmark result
/// [30, 10] -> 33
/// [1000, 200] -> 331
///
/// (batch, input size)
/// polynomial regression

impl Sim {
    pub fn new(targets: Vec<Var>) -> Self {
        Sim {
            mem_budget: 0,
            mem_used: 0,
            peak_mem_used: 0,
            elapsed_time: Duration::from_millis(0),
            targets,
            resolved: HashSet::new(),
        }
    }

    pub fn with_budget(targets: Vec<Var>, mem_budget: usize) -> Self {
        Sim {
            mem_budget,
            mem_used: 0,
            peak_mem_used: 0,
            elapsed_time: Duration::from_millis(0),
            targets,
            resolved: HashSet::new(),
        }
    }

    fn build_depmap(&self) -> HashMap<Var, HashSet<Var>> {
        let mut depmap = HashMap::<Var, HashSet<Var>>::new();

        // Working stack
        let mut stack = Vec::<Var>::new();

        // Simple DFS search
        for target in self.targets.iter() {
            stack.clear();
            stack.push(target.clone());

            while !stack.is_empty() {
                let var = stack.pop().unwrap();
                let node = var.node();

                if let Some(ref op) = node.origin {
                    for in_var in op.input() {
                        if !depmap.contains_key(&in_var) {
                            stack.push(in_var.clone());
                        }

                        depmap.entry(in_var.clone()).or_insert(HashSet::new());
                        depmap.get_mut(&in_var).unwrap().insert(var.clone());
                    }
                }
            }
        }
        depmap
    }

    fn free_dep(&mut self, x: &Var, depmap: &HashMap<Var, HashSet<Var>>) {
        let deps = depmap.get(x).unwrap();

        let is_nodep = deps.iter().all(|v| v.is_evaluated());

        if is_nodep {
            self.resolved.remove(x);
            self.free_mem(x.shape().size());
            x.node_mut().free_data();
        }
    }

    // evaluate vars
    pub fn start(&mut self) {
        let start_time = Instant::now();

        // default work stack
        let mut stack = Vec::<Var>::new();
        // initialize stack with target variables

        let mut tt = self.targets.clone();
        tt.reverse();
        stack.extend(tt);

        // depmap
        let depmap = self.build_depmap();

        let mut iterations = 0;

        while !stack.is_empty() {
            iterations += 1;

            if iterations % 1000 == 0 {
                // println!(
                //     "[{}] stack size: {}, elapsed time: {} sec",
                //     iterations,
                //     stack.len(),
                //     start_time.elapsed().as_millis() as f32 / 1000.0
                // );
            }

            //self.collect_garbage();
            let var = stack.last().cloned().unwrap();
            let mut resolved = false;

            // if not evaluated...
            if !var.is_evaluated() {
                let mut node = var.node_mut();

                // must unwrap, as a node must have either data or parent.

                let op = node
                    .origin
                    .as_ref()
                    .expect("some parameters are not initialized.");

                let unevaluated = op
                    .input()
                    .into_iter()
                    .filter(|v| !v.is_evaluated())
                    .collect::<Vec<Var>>();

                // ready to evaluate!
                if unevaluated.is_empty() {
                    let inputs = op.input();

                    inputs.iter().for_each(|v| {
                        let s = v.shape().size();
                        if self.resolved.insert(v.clone()) {
                            self.alloc_mem(s);
                        }
                    });

                    // exceeds mem budget?
                    //if op.mem_req() > self.mem_budget {
                    // Let's start with something basic.

                    // collect garbage every iteration

                    // ... and move on to the more sophisticated one, only when required.
                    // self.greedy_drop(&in_vars, parent.op.mem_req());
                    //}

                    // do some runtime profiling
                    let timer = Instant::now();

                    let mem_size = node.shape.size();

                    // check mem budget
                    if self.mem_used + mem_size > self.mem_budget && self.mem_budget != 0 {
                        self.greedy_drop(&inputs, self.mem_used + mem_size - self.mem_budget);
                    }

                    //self.alloc_mem(mem_size);

                    let profile = RuntimeProfile {
                        mem_store: mem_size,
                        call_time: Duration::from_secs(1), // static call time
                    };
                    // fill in the null tensor
                    node.data = Some(Tensor::null());
                    node.runtime = Some(profile);
                    if self.resolved.insert(var.clone()) {
                        self.alloc_mem(mem_size);
                    }

                    stack.pop();

                    resolved = true;
                    // clear possible garbage
                } else {
                    stack.extend(unevaluated);
                }
            }
            // already evaluated
            // possible scenario is that user already evaluated one of its ancestor nodes,
            // (i.e., printing loss value before gradient evaluation)
            else {
                if self.resolved.insert(var.clone()) {
                    self.alloc_mem(var.shape().size());
                }
                stack.pop();
            }

            // clear intermediate dependencies.
            if resolved {
                let var_node = var.node();
                let origin = var_node.origin.as_ref().unwrap();

                origin.input().iter().for_each(|v| {
                    if !v.is_leaf() && !self.targets.contains(&v) {
                        self.free_dep(&v, &depmap);
                    }
                })
            }
        }

        self.elapsed_time = start_time.elapsed();
        println!("elapsed time: {} sec", self.elapsed_time.as_secs());
        println!("total iterations: {}", iterations);
    }

    pub fn clear_mem(&mut self) {
        self.resolved.iter().for_each(|v| {
            if !v.is_leaf() {
                v.node_mut().free_data();
            }
        })
    }

    // Garbage collector for variable graph evaluation
    // Clear out inner data of all variable nodes with no dependencies with target variables.
    fn collect_garbage(&mut self) {
        // Variable nodes that ARE dependent to the target variable nodes.
        let mut required: HashSet<Var> = HashSet::new();

        // Working stack
        let mut stack = Vec::<Var>::new();

        // Simple DFS search

        for target in self.targets.iter() {
            stack.clear();
            stack.push(target.clone());

            while !stack.is_empty() {
                let var = stack.pop().unwrap();
                let node = var.node();
                //println!("ss {}", stack.len());

                //if !var.is_evaluated() {
                // if this var is new to the dependency set,
                //if required.insert(var.clone()) {
                // add its ancestors.
                if let Some(ref op) = node.origin {
                    for in_var in op.input() {
                        if required.insert(in_var.clone()) && !in_var.is_evaluated() {
                            stack.push(in_var);
                        }
                    }

                    //stack.extend(op.input());
                    //required.extend(op.input());
                    // add only unevaluated vars to the stack
                    //stack.extend(op.input().into_iter().filter(|v| !v.is_evaluated()))
                }
                //}
                //}
            }
        }

        // keep targets
        //required.extend(self.targets.clone());

        let garbage = self
            .resolved
            .drain_filter(|var| !(required.contains(var) || var.is_leaf()))
            .collect::<Vec<Var>>();

        garbage.iter().for_each(|var| {
            self.free_mem(var.shape().size());
            var.node_mut().free_data();
        });
    }

    // Greedy activation drop
    fn greedy_drop(&mut self, must_keep: &[Var], mem_req: usize) {
        // free ones that have minimum T/M
        let mut mem_freed = 0;

        while mem_freed < mem_req {
            // find variable node with minimum re-computation heuristic (time/space)
            let min_heuristic_var = self
                .resolved
                .iter()
                .filter(|v| !must_keep.contains(v) && !v.is_leaf() && !self.targets.contains(v))
                .min_by_key(|v| {
                    v.node().recompute_heuristic().unwrap() // must unwrap
                })
                .cloned();

            if let Some(var) = min_heuristic_var {
                let mut var_node = var.node_mut();

                if self.resolved.remove(&var) {
                    let mem_size = var_node.shape.size();

                    var_node.free_data();
                    self.free_mem(mem_size);
                    mem_freed += mem_size;
                }
            };
        }
    }

    fn alloc_mem(&mut self, size: usize) {
        self.mem_used += size;
        if self.mem_used > self.peak_mem_used {
            //println!("new peak: {}", self.mem_used * 4 / 1024 / 1024);
            self.peak_mem_used = self.mem_used
        }
    }

    fn free_mem(&mut self, size: usize) {
        self.mem_used -= size;
    }
}
