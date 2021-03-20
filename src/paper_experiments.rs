// experiment #1. memory profile

use crate::autodiff::diff;
use crate::autodiff::ops;
use crate::autodiff::ops::loss::softmax_cross_entropy;
use crate::autodiff::sim::Sim;
use crate::autodiff::var::{ToVar, Var};
use crate::example_models::bert;
use crate::example_models::bert::{Bert, BertConfig};
use crate::example_models::dcgan::{DcGan, DcGanConfig};
use crate::example_models::densenet::{DenseNet, DenseNetConfig};
use crate::example_models::resnet::{ResNet, ResNetConfig};
use crate::example_models::stacked_lstm::{StackedLstm, StackedLstmConfig};
use crate::layers::normalization::LayerNorm;
use crate::layers::{Parameter, Stackable};
use crate::profile::{test_profile, Profiler};
use crate::tensor::Tensor;
use itertools::Itertools;
use std::fs::File;
use std::io::Write;

// input mock

pub fn f32_to_mibs(f32_arr_size: usize) -> f32 {
    (f32_arr_size / 256) as f32 / 1024.0
}
pub fn mibs_to_f32(mibs: usize) -> usize {
    mibs * 256 * 1024
}

pub fn exp1_memory_profile() {
    fn mock_dcgan(batch_size: usize) -> (Var, Vec<Var>) {
        let dcgan = DcGan::new(DcGanConfig::default());
        dcgan.init();

        // Mock input data
        let input_images = Tensor::from_elem([batch_size, 100, 1, 1], 1.0).to_var();
        let labels = Tensor::from_elem([batch_size, 1], 1.0).to_var();

        // Bert logits (classification results)
        let logits = dcgan.forward(&input_images);

        // Evaluate loss
        let loss = softmax_cross_entropy(&logits, &labels);

        let params = dcgan.params().unwrap();
        let grads = diff(&loss, &params);

        let grad_vec = grads.values().cloned().collect_vec();

        (logits, grad_vec)
    }

    fn mock_stacked_lstm(batch_size: usize) -> (Var, Vec<Var>) {
        let stacked_lstm = StackedLstm::new(StackedLstmConfig::d5());
        stacked_lstm.init();

        // Mock input data
        let input_images = Tensor::from_elem([batch_size, 32, 512], 1.0).to_var();
        let labels = Tensor::from_elem([1, 10], 1.0).to_var();

        // Bert logits (classification results)
        let logits = stacked_lstm.forward(&input_images);

        println!("{}, {}", logits.shape(), labels.shape());

        // Evaluate loss
        let loss = softmax_cross_entropy(&logits, &labels);

        let params = stacked_lstm.params().unwrap();
        let grads = diff(&loss, &params);

        let grad_vec = grads.values().cloned().collect_vec();

        (logits, grad_vec)
    }

    fn mock_resnet(batch_size: usize) -> (Var, Vec<Var>) {
        let resnet = ResNet::new(ResNetConfig::d18());
        resnet.init();

        // Mock input data
        let input_images = Tensor::from_elem([batch_size, 3, 128, 128], 1.0).to_var();
        let labels = Tensor::from_elem([batch_size, 10], 1.0).to_var();

        // Bert logits (classification results)
        let logits = resnet.forward(&input_images);

        // Evaluate loss
        let loss = softmax_cross_entropy(&logits, &labels);

        let params = resnet.params().unwrap();
        let grads = diff(&loss, &params);

        let grad_vec = grads.values().cloned().collect_vec();

        (logits, grad_vec)
    }

    fn mock_densenet(batch_size: usize) -> (Var, Vec<Var>) {
        let densenet = DenseNet::new(DenseNetConfig::d121());
        densenet.init();

        // Mock input data
        let input_images = Tensor::from_elem([batch_size, 3, 128, 128], 1.0).to_var();
        let labels = Tensor::from_elem([batch_size, 10], 1.0).to_var();

        // Bert logits (classification results)
        let logits = densenet.forward(&input_images);

        // Evaluate loss
        let loss = softmax_cross_entropy(&logits, &labels);

        let params = densenet.params().unwrap();
        let grads = diff(&loss, &params);

        let grad_vec = grads.values().cloned().collect_vec();

        (logits, grad_vec)
    }

    fn mock_bert(batch_size: usize) -> (Var, Vec<Var>) {
        let bert = Bert::new(BertConfig::base());
        bert.init();

        // Mock input data
        let token_ids = vec![vec![0; 512]; batch_size];
        let attn_mask = Tensor::from_elem([batch_size, 512], 1.0).to_var();
        let labels = Tensor::from_elem([batch_size, 2], 1.0).to_var();

        // Bert logits (classification results)
        let logits = bert.forward(&token_ids, &attn_mask);

        // Evaluate loss
        let loss = softmax_cross_entropy(&logits, &labels);

        let params = bert.params().unwrap();
        let grads = diff(&loss, &params);

        let grad_vec = grads.values().cloned().collect_vec();

        (logits, grad_vec)
    }

    let batch_size = 8;
    //
    // let resnet = mock_resnet(batch_size);
    // let densenet = mock_densenet(batch_size);
    // let bert = mock_bert(batch_size);
    // let stacked_lstm = mock_stacked_lstm(batch_size);
    // let dcgan = mock_dcgan(batch_size);
    //
    // let models = [resnet, densenet, bert, stacked_lstm, dcgan];
    // let model_names = ["resnet", "densenet", "bert", "stacked_lstm", "dcgan"];
    //
    // let batch_sizes = [1, 2, 4, 8, 16, 32, 64];
    //
    // for i in 0..5 {
    //     let model = &models[i];
    //     let model_name = &model_names[i];
    //
    //     if i != 3 {
    //         continue;
    //     }
    //
    //     let mut sim = Sim::new(model.1.clone());
    //     sim.start();
    //     sim.clear_mem();
    //
    //     println!("[model {}] normal peak: {}", model_name, sim);
    //
    //     let mut budget = (sim.peak_mem_used as f32 / 1.5) as usize;
    //     let iter_threshold = 10000000;
    //
    //     loop {
    //         println!("trying: {}", f32_to_mibs(budget));
    //
    //         let mut sim = Sim::with_budget(model.1.clone(), budget, iter_threshold);
    //         if !sim.start() {
    //             sim.clear_mem();
    //             break;
    //         }
    //         sim.clear_mem();
    //         budget -= mibs_to_f32(20);
    //     }
    //
    //     println!("[model {}] min peak  : {}", model_name, f32_to_mibs(budget));
    // }

    let mut profiler = test_profile();

    let bert = mock_densenet(4);
    let budget = 4000;

    let mut sim = Sim::with_budget(&mut profiler, bert.1.clone(), mibs_to_f32(budget), 1000000);
    sim.start();
    let mut kv = String::new();
    let mut rv = String::new();
    let mut tv = String::new();
    let mut ev = String::new();

    println!("{}", sim.calltrace.len());

    for (k, r, t, e) in sim.calltrace.iter() {
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

    let mut file = File::create("ep.csv").unwrap();
    file.write_all(kv.as_ref());

    //
    // let batches = [16];
    // let budgets = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300];
    //
    // let models = [
    //     //mock_resnet,
    //     //mock_densenet,
    //     //mock_bert,
    //     //mock_stacked_lstm,
    //     mock_dcgan,
    // ];
    //
    // for &model in models.iter() {
    //     for &batch in batches.iter() {
    //         for &budget in budgets.iter().rev() {
    //             let bert = model(batch);
    //             let mut sim =
    //                 Sim::with_budget(&mut profiler, bert.1.clone(), mibs_to_f32(budget), 1000000);
    //             sim.start();
    //             sim.clear_mem();
    //             println!("({}, {}): {}", batch, budget, sim);
    //         }
    //     }
    // }
    //
    //
    //
    // let mut file = File::create("torch_bench.txt").unwrap();
    // file.write_all(profiler.gen_benchmark(2).as_ref());

    // return;
    //
    // for (i, model) in models.iter().enumerate() {
    //     // evaluate grads
    //     // forward pass
    //     let model_name = model_names[i];
    //
    //     if model_name != "bert" {
    //         continue;
    //     }
    //
    //     //let min_budget = min_budgets[i];
    //
    //     println!("evaluating {}", model_name);
    //
    //     let mut sim = Sim::new(vec![model.0.clone()]);
    //     sim.start();
    //     sim.clear_mem();
    //
    //     println!("    - forward: {}", sim);
    //
    //     let mut sim = Sim::new(model.1.clone());
    //     sim.start();
    //     sim.clear_mem();
    //
    //     println!("    - backward: {}", sim);
    // }

    //return;

    // 7GB
    // for i in 1..2 {
    //     let mem_budget_in_mibs = 3000;
    //
    //     let mut sim = Sim::with_budget(grad_vec.clone(), &loss, mibs_to_f32(mem_budget_in_mibs));
    //     sim.start();
    //     sim.clear_mem();
    //
    //     println!(
    //         "backward pass (budget: {} MB): {} MB",
    //         mem_budget_in_mibs,
    //         f32_to_mibs(sim.peak_mem_used)
    //     );
    // }
}

// experiment #2. minimum memory requirements

pub fn exp2_min_mem_req() {}

// experiment #3. time-memory trade-off analysis

pub fn exp3_cost_analysis() {}

// experiment #4 dynamic memory budget

pub fn exp4_dynamic_mem_budget() {}
