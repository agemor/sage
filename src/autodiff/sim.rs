// computational costs of each function
// profiled manually

// backward graph

use crate::autodiff::var::RuntimeProfile;
use crate::autodiff::Var;
use crate::paper_experiments::f32_to_mibs;
use crate::profile::Profiler;
use crate::tensor::Tensor;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};
use std::{cmp, fmt};

// Variable evaluation session
pub struct Sim<'a> {
    pub mem_budget: usize,
    pub mem_used: usize,
    pub peak_mem_used: usize,

    pub model_mem: usize,
    pub total_iter: usize,
    pub elapsed_time: f32,
    pub comp_time: usize,
    pub energy_use: usize,

    pub targets: Vec<Var>,
    pub iter_threshold: usize,

    pub global_lock: HashSet<Var>,
    pub once_evicted: HashSet<Var>,
    pub resolved: HashSet<Var>,

    pub profiler: &'a mut Profiler,
    pub memtrace: Vec<(usize, usize, bool)>,
    pub calltrace: Vec<(bool, bool, usize, usize)>,
}

impl<'a> fmt::Display for Sim<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mem_budget: {}, model_mem: {} MB, peak_mem: {} MB, total_iter: {}, elapsed_time: {} sec, comp_time: {}",
            f32_to_mibs(self.mem_budget),
            f32_to_mibs(self.model_mem),
            f32_to_mibs(self.peak_mem_used),
            self.total_iter,
            self.elapsed_time,
            self.comp_time
        )
    }
}

///
/// benchmark result
/// [30, 10] -> 33
/// [1000, 200] -> 331
///
/// (batch, input size)
/// polynomial regression

impl<'a> Sim<'a> {
    pub fn new(profiler: &'a mut Profiler, targets: Vec<Var>) -> Self {
        Sim {
            mem_budget: 0,
            mem_used: 0,
            peak_mem_used: 0,
            elapsed_time: 0.0,
            model_mem: 0,
            total_iter: 0,
            comp_time: 0,
            energy_use: 0,
            targets,
            iter_threshold: 0,
            global_lock: HashSet::new(),
            once_evicted: HashSet::new(),
            resolved: HashSet::new(),
            profiler,
            memtrace: Vec::new(),
            calltrace: Vec::new(),
        }
    }

    pub fn with_budget(
        profiler: &'a mut Profiler,
        targets: Vec<Var>,
        mem_budget: usize,
        iter_threshold: usize,
    ) -> Self {
        Sim {
            mem_budget,
            mem_used: 0,
            peak_mem_used: 0,
            elapsed_time: 0.0,
            model_mem: 0,
            total_iter: 0,
            comp_time: 0,
            energy_use: 0,
            targets,
            iter_threshold,
            global_lock: HashSet::new(),
            once_evicted: HashSet::new(),
            resolved: HashSet::new(),
            profiler,
            memtrace: Vec::new(),
            calltrace: Vec::new(),
        }
    }

    fn build_subset(target: &Var) -> HashSet<Var> {
        let mut res = HashSet::new();

        let mut stack = Vec::<Var>::new();
        stack.push(target.clone());

        while !stack.is_empty() {
            let var = stack.pop().unwrap();
            res.insert(var.clone());
            let node = var.node();
            if let Some(ref op) = node.origin {
                for in_var in op.input() {
                    if !res.contains(&in_var) {
                        stack.push(in_var.clone());
                    }
                }
            }
        }
        res
    }

    fn build_depmap(&mut self) -> HashMap<Var, HashSet<Var>> {
        let mut depmap = HashMap::<Var, HashSet<Var>>::new();

        // Working stack
        let mut stack = Vec::<Var>::new();

        let mut total_grad_mem = 0;
        let mut total_model_mem = 0;

        //println!(" * analyzing computational graph...");

        // Simple DFS search
        for target in self.targets.iter() {
            stack.clear();
            stack.push(target.clone());

            while !stack.is_empty() {
                let var = stack.pop().unwrap();

                // should keep
                if var.is_leaf() {
                    total_model_mem += var.shape().size();
                }

                if self.targets.contains(&var) {
                    total_grad_mem += var.shape().size();
                }

                let node = var.node();

                if let Some(ref op) = node.origin {
                    for in_var in op.input() {
                        if !depmap.contains_key(&in_var) {
                            stack.push(in_var.clone());
                        }

                        depmap.entry(in_var.clone()).or_insert_with(HashSet::new);
                        depmap.get_mut(&in_var).unwrap().insert(var.clone());
                    }
                }
            }
        }
        self.model_mem = total_model_mem;

        //println!("   - model mem: {} MB", f32_to_mibs(total_model_mem));
        //println!("   - grad mem: {} MB", f32_to_mibs(total_grad_mem));

        depmap
    }

    fn free_dep(&mut self, x: &Var, depmap: &HashMap<Var, HashSet<Var>>) {
        let deps = depmap.get(x).unwrap();

        let is_nodep = deps.iter().all(|v| v.is_evaluated());

        if is_nodep && self.resolved.remove(x) {
            self.free_mem(x.shape().size());
            x.node_mut().free_data();
        }
    }

    fn real_mem_use(&self) -> usize {
        if self.resolved.iter().any(|v| !v.is_evaluated()) {
            panic!("something wrong");
        }

        self.resolved
            .iter()
            .fold(0, |acc, v| acc + v.shape().size())
    }

    // evaluate vars
    pub fn start(&mut self) -> bool {
        let start_time = Instant::now();
        let mut comp_time: usize = 0;
        let mut recomp_time: usize = 0;
        let mut energy_use: usize = 0;

        // dynamic budget

        let budget_plan = [
            0.7, 0.7, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7,
            1.3, 1.3, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
        ];

        // default work stack
        let mut stack = Vec::<Var>::new();
        let mut visit_planned = HashSet::<Var>::new();
        // initialize stack with target variables

        stack.extend(self.targets.clone());
        visit_planned.extend(self.targets.clone());

        // println!("[session started]");

        // depmap
        let depmap = self.build_depmap();

        let mut iterations = 0;

        //println!(" * simulating computational graph...");

        while !stack.is_empty() {
            iterations += 1;

            let bpi = (iterations / (150) % budget_plan.len();
            let current_budget = (self.mem_budget as f32 * budget_plan[bpi]) as usize;

            if iterations > self.iter_threshold && self.iter_threshold != 0 {
                return false;
            }

            // if iterations % 100 == 0 {
            //     println!(
            //         "[{}] stack size: {}, elapsed time: {} sec",
            //         iterations,
            //         stack.len(),
            //         start_time.elapsed().as_millis() as f32 / 1000.0
            //     );
            // }

            // let mem_use_real = self.real_mem_use();
            //
            // if mem_use_real != self.mem_used {
            //     panic!("wrong mem use");
            // }

            //self.collect_garbage();
            let var = stack.last().cloned().unwrap();
            let mut resolved = false;
            let mut is_recomputation = false;

            // if not evaluated...
            if !var.is_evaluated() {
                let node = var.node();

                // must unwrap, as a node must have either data or parent.

                let op = node
                    .origin
                    .as_ref()
                    .expect("some parameters are not initialized.");

                // acquire global lock for this variable node
                self.global_lock.extend(op.input());
                //self.global_lock.insert(op.output().unwrap());

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
                    //println!("current node : {}", op.debug_info());

                    if self.mem_used + mem_size > current_budget && self.mem_budget != 0 {
                        let mut must_keep = Vec::new();
                        // self-closure
                        if let Some(depmap) = depmap.get(&var) {
                            for v in depmap.iter() {
                                let node = v.node();
                                if let Some(op) = &node.origin {
                                    must_keep.extend(op.input());
                                }
                            }
                        }

                        must_keep.extend(inputs.clone());

                        // println!("current node : {}", op.debug_info());
                        // println!(
                        //     "mem use: {} MB, must_keep: {}",
                        //     f32_to_mibs(mem_size),
                        //     must_keep.len()
                        // );

                        recomp_time +=
                            self.greedy_drop(&must_keep, self.mem_used + mem_size - current_budget);
                    }

                    if self.resolved.insert(var.clone()) {
                        /////////////////////// AADDDDD BBENCCCCHHH!!

                        //op.add_bench(&mut self.profiler);

                        let di = op.debug_info(&self.profiler);
                        comp_time += di.comp_time;
                        energy_use += (di.comp_time as f32 * di.energy_factor) as usize;

                        is_recomputation = self.once_evicted.contains(&var);

                        let is_energy_intensive = di.energy_factor > 1.0;

                        self.calltrace.push((
                            is_energy_intensive,
                            is_recomputation,
                            comp_time,
                            energy_use,
                        ));

                        self.alloc_mem(mem_size);
                    }

                    let v = stack.pop().unwrap();
                    visit_planned.remove(&v);

                    // release global lock
                    inputs.iter().for_each(|v| {
                        self.global_lock.remove(v);
                    });

                    resolved = true;

                    //self.alloc_mem(mem_size);

                    drop(node);

                    let mut node = var.node_mut();
                    let mem_size = node.shape.size();

                    let profile = RuntimeProfile {
                        mem_store: mem_size,
                        call_time: Duration::from_secs(1), // static call time
                    };

                    // fill in the null tensor
                    node.data = Some(Tensor::null());
                    node.runtime = Some(profile);

                    // clear possible garbage
                } else {
                    //unevaluated.sort_by_key(|v| v.shape().size());

                    for v in unevaluated {
                        if visit_planned.contains(&v) {
                            // re-prioritize stack
                            let (i, _) = stack.iter().find_position(|&v| v.eq(&v)).unwrap();
                            stack.remove(i);
                        }
                        stack.push(v);
                    }

                    // stack.extend(unevaluated);
                }
            }
            // already evaluated
            // possible scenario is that user already evaluated one of its ancestor nodes,
            // (i.e., printing loss value before gradient evaluation)
            else {
                if self.resolved.insert(var.clone()) {
                    self.alloc_mem(var.shape().size());
                }

                let v = stack.pop().unwrap();
                visit_planned.remove(&v);

                //stack.pop();
                resolved = true;
            }

            // clear intermediate dependencies.
            if resolved {
                //visit_planned.remove(&var);

                let var_node = var.node();
                let origin = var_node.origin.as_ref().unwrap();

                origin.input().iter().for_each(|v| {
                    if !v.is_leaf() && !self.targets.contains(&v) {
                        self.free_dep(&v, &depmap);
                    }
                })
            }
            self.memtrace
                .push((self.mem_used, current_budget, is_recomputation));

            // if self.mem_used > self.mem_budget && self.mem_budget != 0 {
            //     self.greedy_drop(&[var], self.mem_used - self.mem_budget);
            // }
        }
        //
        // if self.mem_budget == 0 {
        //     println!("   - budget mem: unlimited");
        // } else {
        //     println!("   - budget mem: {} MB", f32_to_mibs(self.mem_budget));
        // }
        //
        // println!("   - peak mem: {} MB", f32_to_mibs(self.peak_mem_used));
        //
        // println!("   - comp time: {} millis", comp_time / 1024 / 1024 / 1024);
        self.comp_time = comp_time;
        self.energy_use = energy_use;
        self.total_iter = iterations;
        self.elapsed_time = start_time.elapsed().as_millis() as f32 / 1000.0;

        // println!(
        //     "[session closed] total iterations: {}, elapsed time: {} sec",
        //     iterations,
        //     start_time.elapsed().as_millis() as f32 / 1000.0
        // );

        return true;
    }

    // Greedy activation drop
    fn greedy_drop(&mut self, must_keep: &[Var], mem_req: usize) -> usize {
        // free ones that have minimum T/M
        let mut mem_freed = 0;
        let mut iterations = 0;

        let rank = must_keep[0].rank();
        let now = Instant::now();
        let mut recomp_time = 0;

        fn dist(a: usize, b: usize) -> usize {
            if a > b {
                a - b
            } else {
                b - a
            }
        }

        while mem_freed < mem_req {
            iterations += 1;
            // find variable node with minimum re-computation heuristic (time/space)
            let min_heuristic_var = self
                .resolved
                .iter()
                .filter(|v| !self.global_lock.contains(v) && !v.is_leaf())
                //.max_by_key(|v| v.shape().size())
                .min_by_key(|v| {
                    (v.node().recompute_heuristic(&self.profiler).unwrap() * 100000.0) as usize
                })
                .cloned();

            // ! must_keep.contains(v)()
            //&& !self.targets.contains(v)
            // && self.forward_subset.contains(v)

            if let Some(var) = min_heuristic_var {
                // must unwrap
                //println!("freed node : {}", var.debug_info().unwrap());

                if self.resolved.remove(&var) {
                    recomp_time += var.debug_info(&self.profiler).unwrap().comp_time;

                    let mut var_node = var.node_mut();

                    let mem_size = var_node.shape.size();
                    self.once_evicted.insert(var.clone());
                    var_node.free_data();
                    self.free_mem(mem_size);
                    mem_freed += mem_size;
                }
            } else {
                let mm = self
                    .resolved
                    .iter()
                    .filter(|v| {
                        self.global_lock.contains(v) || v.is_leaf()
                        //must_keep.contains(v) || v.is_leaf()
                        //   || self.targets.contains(v)
                        //   || !(self.forward_subset.contains(v))
                    })
                    .fold(0, |acc, v| acc + v.shape().size());

                panic!(
                    "cannot free more! occupied: {} MB, requested: {} MB, fixed: {} MB",
                    f32_to_mibs(self.mem_used),
                    f32_to_mibs(mem_req),
                    f32_to_mibs(mm)
                );
            }
        }

        // println!(
        //     "  *   [gc] requested mem: {} MB, freed: {} MB  (affected: {}, time: {} sec)",
        //     f32_to_mibs(mem_req),
        //     f32_to_mibs(mem_freed),
        //     iterations,
        //     (now.elapsed().as_millis() as f32) / 1000.0,
        // );

        recomp_time
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

    pub fn clear_mem(&mut self) {
        self.resolved.iter().for_each(|v| {
            if !v.is_leaf() {
                v.node_mut().free_data();
            }
        })
    }

    fn alloc_mem(&mut self, size: usize) {
        self.mem_used += size;
        if self.mem_used > self.peak_mem_used {
            //println!("new peak: {}", self.mem_used/ 1024 / 1024);
            self.peak_mem_used = self.mem_used
        }
    }

    fn free_mem(&mut self, size: usize) {
        self.mem_used -= cmp::min(size, self.mem_used);
    }

    pub fn save_calltrace(&self, file:&str) {
        let mut kv = String::new();
        let mut rv = String::new();
        let mut tv = String::new();
        let mut ev = String::new();

        println!("{}", self.calltrace.len());

        for (k, r, t, e) in self.calltrace.iter() {
            kv.push(if *k { '1' } else { '0' });
            kv.push(',');

            rv.push(if *r { '1' } else { '0' });
            rv.push(',');

            tv.push_str(&t.to_string());
            tv.push(',');

            ev.push_str(&e.to_string());
            ev.push(',');
        }
        kv.push('\n');
        kv.push_str(&rv);
        kv.push('\n');
        kv.push_str(&tv);
        kv.push('\n');
        kv.push_str(&ev);

        let mut file = File::create(file).unwrap();
        file.write_all(kv.as_ref());
    }

    pub fn save_memtrace(&self, file:&str) {
        let mut uv = String::new();
        let mut bv = String::new();
        let mut rv = String::new();

        println!("{}", self.calltrace.len());

        for (u, b, r) in self.memtrace.iter() {
            uv.push_str(&u.to_string());
            uv.push(',');

            bv.push_str(&b.to_string());
            bv.push(',');

            rv.push(if *r { '1' } else { '0' });
            rv.push(',');
        }
        uv.push('\n');
        uv.push_str(&bv);
        uv.push('\n');
        uv.push_str(&rv);

        let mut file = File::create(file).unwrap();
        file.write_all(uv.as_ref());
    }
}
