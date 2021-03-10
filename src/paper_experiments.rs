// experiment #1. memory profile

use crate::autodiff::diff;
use crate::autodiff::ops;
use crate::autodiff::sim::Sim;
use crate::example_models::bert;
use crate::example_models::bert::{Bert, BertConfig};

// input mock

pub fn exp1_memory_profile() {
    let bert = Bert::new(BertConfig::base());

    let logits = bert.forward();

    let loss = ops::softmax_cross_entropy(&logits, &labels);

    // forward pass
    Sim::with_budget(vec![logits], 0);

    // backward pass

    let params = model.params().unwrap();
    let grads = diff(&loss, &params);
}

// experiment #2. minimum memory requirements

pub fn exp2_min_mem_req() {}

// experiment #3. time-memory trade-off analysis

pub fn exp3_cost_analysis() {}

// experiment #4 dynamic memory budget

pub fn exp4_dynamic_mem_budget() {}
