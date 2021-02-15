use crate::autodiff::{RuntimeProfile, Var};
use crate::tensor;
use crate::tensor::Tensor;
use std::cell::Ref;
use std::collections::HashSet;
use std::ops::Deref;
use std::time::Instant;

// Variable evaluation session
pub struct Session {
    mem_budget: usize,

    targets: Vec<Var>,

    resolved: HashSet<Var>,
}

impl Session {
    pub fn with_budget(targets: Vec<Var>, mem_budget: usize) -> Self {
        Session {
            mem_budget,
            targets,
            resolved: HashSet::new(),
        }
    }

    // evaluate vars
    pub fn eval(&mut self) {
        // default work stack
        let mut stack = Vec::<Var>::new();

        // initialize stack with target variables
        stack.extend(self.targets.clone());

        while !stack.is_empty() {
            let var = stack.last().cloned().unwrap();

            let mut var_node = var.node_mut();

            // if not evaluated...
            if !var.is_evaluated() {
                // must unwrap, as a node must have either data or parent.
                let parent = var_node.parent.as_ref().unwrap();

                let unevaluated = parent
                    .input_vars()
                    .into_iter()
                    .filter(|v| !v.is_evaluated())
                    .collect::<Vec<Var>>();

                // ready to evaluate!
                if unevaluated.is_empty() {
                    let in_vars = parent.input_vars();

                    // exceeds mem budget?
                    if parent.op.mem_req() > self.mem_budget {
                        // Let's start with something basic.
                        self.collect_garbage();

                        // ... and move on to the more sophisticated one, only when required.
                        self.greedy_drop(&in_vars, parent.op.mem_req());
                    }

                    let in_tensors = in_vars
                        .iter()
                        .map(|v| v.data_unchecked())
                        .collect::<Vec<Ref<Tensor>>>();

                    let in_tensors2 = in_tensors
                        .iter()
                        .map(|v| v.deref())
                        .collect::<Vec<&Tensor>>();

                    // do some runtime profiling
                    let timer = Instant::now();

                    let out_tensor = parent.op.compute(&in_tensors2);

                    let profile = RuntimeProfile {
                        mem_store: tensor::mem_size(&out_tensor),
                        call_time: timer.elapsed(),
                    };

                    var_node.data = Some(out_tensor);
                    var_node.runtime = Some(profile);

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
                let var_node = var.node();

                if !var.is_evaluated() {
                    // if this var is new to the dependency set,
                    if required.insert(var.clone()) {
                        // add its ancestors.
                        if let Some(ref p) = var_node.parent {
                            stack.extend(p.input_vars())
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
