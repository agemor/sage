// computational costs of each function
// profiled manually

// backward graph

use crate::autodiff::var::RuntimeProfile;
use crate::autodiff::Var;
use std::collections::HashSet;
use std::time::Instant;

// Variable evaluation session
pub struct Sim {
    mem_budget: usize,

    targets: Vec<Var>,

    resolved: HashSet<Var>,
}

///
/// benchmark result
/// [30, 10] -> 33
/// [1000, 200] -> 331
///
/// (batch, input size)
/// polynomial regression

impl Sim {
    pub fn with_budget(targets: Vec<Var>, mem_budget: usize) -> Self {
        Sim {
            mem_budget,
            targets,
            resolved: HashSet::new(),
        }
    }

    // evaluate vars
    pub fn start(&mut self) {
        // default work stack
        let mut stack = Vec::<Var>::new();

        // initialize stack with target variables
        stack.extend(self.targets.clone());

        while !stack.is_empty() {
            let var = stack.last().cloned().unwrap();

            // if not evaluated...
            if !var.is_evaluated() {
                let mut node = var.node_mut();

                // must unwrap, as a node must have either data or parent.
                let op = node.origin.as_ref().unwrap();

                let unevaluated = op
                    .input()
                    .into_iter()
                    .filter(|v| !v.is_evaluated())
                    .collect::<Vec<Var>>();

                // ready to evaluate!
                if unevaluated.is_empty() {
                    // exceeds mem budget?
                    if op.mem_req() > self.mem_budget {
                        // Let's start with something basic.
                        //self.collect_garbage();

                        // ... and move on to the more sophisticated one, only when required.
                        // self.greedy_drop(&in_vars, parent.op.mem_req());
                    }

                    // do some runtime profiling
                    let timer = Instant::now();

                    let data = op.compute();

                    let profile = RuntimeProfile {
                        mem_store: data.mem_size(),
                        call_time: timer.elapsed(),
                    };

                    node.data = Some(data);
                    node.runtime = Some(profile);

                    self.resolved.insert(var.clone());

                    stack.pop();
                } else {
                    stack.extend(unevaluated);
                }
            }
            // already evaluated
            // possible scenario is that user already evaluated one of its ancestor nodes,
            // (i.e., printing loss value before gradient evaluation)
            else {
                self.resolved.insert(var.clone());
                stack.pop();
            }
        }
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

                if !var.is_evaluated() {
                    // if this var is new to the dependency set,
                    if required.insert(var.clone()) {
                        // add its ancestors.
                        if let Some(ref op) = node.origin {
                            stack.extend(op.input())
                        }
                    }
                }
            }
        }

        // keep targets
        required.extend(self.targets.clone());

        let garbage = self
            .resolved
            .drain_filter(|var| !required.contains(var))
            .collect::<Vec<Var>>();

        garbage.iter().for_each(|var| {
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
                .filter(|v| !must_keep.contains(v))
                .min_by_key(|v| {
                    v.node().recompute_heuristic().unwrap() // must unwrap
                })
                .cloned();

            if let Some(var) = min_heuristic_var {
                let mut var_node = var.node_mut();

                var_node.free_data();

                let runtime = var_node.runtime.as_ref().unwrap();

                mem_freed += runtime.mem_store;
                self.resolved.remove(&var);
            };
        }
    }
}
