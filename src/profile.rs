use crate::tensor::shape::ToShape;
use std::collections::{HashMap, HashSet};

pub fn torch_var<S>(name: &str, shape: S) -> String
where
    S: ToShape,
{
    let shape = shape.to_shape(0);
    format!("{} = torch.randn([{}]).cuda()\n", name, shape.to_string())
}

pub struct Profiler {
    comp_time: HashMap<String, usize>,
    benchmark: HashMap<String, (String, String)>,
}

impl Profiler {
    pub fn new() -> Self {
        Profiler {
            comp_time: HashMap::new(),
            benchmark: HashMap::new(),
        }
    }

    pub fn add_benchmark(&mut self, uid: &str, prep_code: String, exec_code: String) {
        self.benchmark
            .insert(uid.to_string(), (prep_code, exec_code));
    }

    pub fn comp_time(&self, uid: &str) -> Option<usize> {
        self.comp_time.get(uid).cloned()
    }

    pub fn gen_benchmark(&self, repeat: usize) -> String {
        // step 1. profiler code

        let mut prep = String::new();
        let mut exec = String::new();

        let indent = "    ";

        prep.push_str("import timeit\n");
        prep.push_str("import torch\n");

        exec.push_str("benchmark = {}\n");
        exec.push_str(&format!("for i in range({}):\n", repeat));

        for (uid, (prep_code, exec_code)) in self.benchmark.iter() {
            prep.push_str(prep_code);
            prep.push_str("\n");

            // timeit.timeit(lambda: {exec_code}, number=1)
            exec.push_str(indent);
            exec.push_str(&format!(
                "t = timeit.timeit(lambda: {}, number=1)\n",
                exec_code
            ));
            exec.push_str(indent);
            exec.push_str(&format!("benchmark[{}].setdefault(0) \n", uid));
            exec.push_str(indent);
            exec.push_str(&format!("benchmark[{}] += t\n", uid));
        }

        exec.push_str("rust_code = '['\n");
        exec.push_str("for uid in benchmark:\n");
        exec.push_str(indent);
        exec.push_str("rust_code += '(' + uid + ', ' + (benchmark[uid] / repeat) + '), '\n");
        exec.push_str("rust_code += ']'\n");
        exec.push_str("print(rust_code)'\n");

        prep.push_str("\n");
        prep.push_str(&exec);

        prep
    }
}

macro_rules! map(
    { $($key:expr => $value:expr),+ } => {
        {
            let mut m = ::std::collections::HashMap::new();
            $(
                m.insert($key, $value);
            )+
            m
        }
     };
);
