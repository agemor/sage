// experiment #1. memory profile

use crate::autodiff::diff;
use crate::autodiff::ops;
use crate::autodiff::ops::loss::softmax_cross_entropy;
use crate::autodiff::sim::Sim;
use crate::autodiff::var::{ToVar, Var};
use crate::example_models::bert;
use crate::example_models::bert::{Bert, BertConfig};
use crate::example_models::densenet::{DenseNet, DenseNetConfig};
use crate::layers::normalization::LayerNorm;
use crate::layers::{Parameter, Stackable};
use crate::tensor::Tensor;
use itertools::Itertools;
use crate::example_models::resnet::{ResNet, ResNetConfig};

// input mock

pub fn f32_to_mibs(f32_arr_size: usize) -> f32 {
    (f32_arr_size / 256) as f32 / 1024.0
}
pub fn mibs_to_f32(mibs: usize) -> usize {
    mibs * 256 * 1024
}

pub fn exp1_memory_profile() {

    fn test_resnet(batch_size:usize) {

        let resnet = ResNet::new(ResNetConfig::d50());
        resnet.init();

        // Mock input data
        let input_images = Tensor::from_elem([batch_size, 3, 224, 224], 1.0).to_var();
        let labels = Tensor::from_elem([batch_size, 10], 1.0).to_var();

        // Bert logits (classification results)
        let logits = resnet.forward(&input_images);

        println!("{} {}", logits.shape(), labels.shape());

        // Evaluate loss
        let loss = softmax_cross_entropy(&logits, &labels);

        let params = resnet.params().unwrap();
        let grads = diff(&loss, &params);

        let grad_vec = grads.values().cloned().collect_vec();

        // evaluate grads
        // forward pass
        let mut sim = Sim::new(vec![logits], &loss);
        sim.start();
        sim.clear_mem();

        println!("forward (unlimited): {} MB", f32_to_mibs(sim.peak_mem_used));

        // backward pass

        let mut sim = Sim::new(grad_vec.clone(), &loss);
        sim.start();
        sim.clear_mem();

        println!(
            "backward (unlimited): {} MB",
            f32_to_mibs(sim.peak_mem_used)
        );

        //return;
        // 7GB
        for i in 1..2 {
            let mem_budget_in_mibs = 3000;

            let mut sim =
                Sim::with_budget(grad_vec.clone(), &loss, mibs_to_f32(mem_budget_in_mibs));
            sim.start();
            sim.clear_mem();

            println!(
                "backward (budget: {} MB): {} MB",
                mem_budget_in_mibs,
                f32_to_mibs(sim.peak_mem_used)
            );
        }

    }

    fn test_densenet(batch_size: usize) {
        let densenet = DenseNet::new(DenseNetConfig::d121());
        densenet.init();

        // Mock input data
        let input_images = Tensor::from_elem([batch_size, 3, 224, 224], 1.0).to_var();
        let labels = Tensor::from_elem([batch_size, 10], 1.0).to_var();

        // Bert logits (classification results)
        let logits = densenet.forward(&input_images);

        println!("{} {}", logits.shape(), labels.shape());

        // Evaluate loss
        let loss = softmax_cross_entropy(&logits, &labels);

        let params = densenet.params().unwrap();
        let grads = diff(&loss, &params);

        let grad_vec = grads.values().cloned().collect_vec();

        // evaluate grads
        // forward pass
        let mut sim = Sim::new(vec![logits], &loss);
        sim.start();
        sim.clear_mem();

        println!("forward (unlimited): {} MB", f32_to_mibs(sim.peak_mem_used));

        // backward pass

        let mut sim = Sim::new(grad_vec.clone(), &loss);
        sim.start();
        sim.clear_mem();

        println!(
            "backward (unlimited): {} MB",
            f32_to_mibs(sim.peak_mem_used)
        );

        //return;
        // 7GB
        for i in 1..2 {
            let mem_budget_in_mibs = 3000;

            let mut sim =
                Sim::with_budget(grad_vec.clone(), &loss, mibs_to_f32(mem_budget_in_mibs));
            sim.start();
            sim.clear_mem();

            println!(
                "backward (budget: {} MB): {} MB",
                mem_budget_in_mibs,
                f32_to_mibs(sim.peak_mem_used)
            );
        }
    }

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
        let mut sim = Sim::new(vec![logits], &loss);
        sim.start();
        sim.clear_mem();

        println!("forward pass (unlimited): {} MB", f32_to_mibs(sim.peak_mem_used));

        // backward pass

        let mut sim = Sim::new(grad_vec.clone(), &loss);
        sim.start();
        sim.clear_mem();

        println!(
            "backward pass (unlimited): {} MB",
            f32_to_mibs(sim.peak_mem_used)
        );
        //return;

        // 7GB
        for i in 1..2 {
            let mem_budget_in_mibs = 1024;

            let mut sim =
                Sim::with_budget(grad_vec.clone(), &loss, mibs_to_f32(mem_budget_in_mibs));
            sim.start();
            sim.clear_mem();

            println!(
                "backward pass (budget: {} MB): {} MB",
                mem_budget_in_mibs,
                f32_to_mibs(sim.peak_mem_used)
            );
        }
    }
    test_resnet(1);
    //test_bert(1, 512);
}

// experiment #2. minimum memory requirements

pub fn exp2_min_mem_req() {}

// experiment #3. time-memory trade-off analysis

pub fn exp3_cost_analysis() {}

// experiment #4 dynamic memory budget

pub fn exp4_dynamic_mem_budget() {}
