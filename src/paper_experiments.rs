// experiment #1. memory profile

use crate::autodiff::diff;
use crate::autodiff::ops;
use crate::autodiff::ops::loss::softmax_cross_entropy;
use crate::autodiff::sim::Sim;
use crate::autodiff::var::{ToVar, Var};
use crate::example_models::bert;
use crate::example_models::bert::{Bert, BertConfig};
use crate::layers::normalization::LayerNorm;
use crate::layers::{Parameter, Stackable};
use crate::tensor::Tensor;
use itertools::Itertools;

// input mock

pub fn bytes_to_megabytes(bytes: usize) -> f32 {
    (bytes / 1024) as f32 / 1024.0
}

pub fn exp1_memory_profile() {
    fn test_bert(batch_size: usize, word_len: usize) {
        let bert = Bert::new(BertConfig::base());
        bert.init();

        // Mock input data
        let token_ids = vec![vec![0; word_len]; batch_size];
        let attn_mask = Tensor::from_elem([batch_size, word_len], 1.0).to_var();
        let labels = Tensor::from_elem([batch_size, 2], 1.0).to_var();

        // Bert logits (classification results)
        let logits = bert.forward(&token_ids, &attn_mask);

        // Evaluate loss
        let loss = softmax_cross_entropy(&logits, &labels);

        let params = bert.params().unwrap();
        let grads = diff(&loss, &params);

        let grad_vec = grads.values().cloned().collect_vec();

        // evaluate grads

        // forward pass
        let mut sim = Sim::new(vec![logits]);
        sim.start();

        println!(
            "forward (unlimited): {} MB",
            bytes_to_megabytes(sim.peak_mem_used * 4)
        );

        // backward pass

        let mut sim = Sim::new(grad_vec.clone());
        sim.start();
        sim.clear_mem();

        println!(
            "backward (unlimited): {} MB",
            bytes_to_megabytes(sim.peak_mem_used * 4)
        );

        // 7GB
        for i in 1..6 {
            let mem_budget = (8 - i) * 256 * 1024 * 1024;

            let mut sim = Sim::with_budget(grad_vec.clone(), mem_budget);
            sim.start();
            sim.clear_mem();

            println!(
                "backward (budget: {} MB): {} MB",
                bytes_to_megabytes(mem_budget * 4),
                bytes_to_megabytes(sim.peak_mem_used * 4)
            );
        }
    }
    test_bert(1, 512);
}

// experiment #2. minimum memory requirements

pub fn exp2_min_mem_req() {}

// experiment #3. time-memory trade-off analysis

pub fn exp3_cost_analysis() {}

// experiment #4 dynamic memory budget

pub fn exp4_dynamic_mem_budget() {}
