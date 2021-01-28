use crate::autodiff::Var;

trait Net {
    fn params(&self) -> &[&Var];
}

struct Linear {
    w: Var,
    b: Var,
}

impl Linear {
    fn new(input: u32, output: u32) -> Self {
        Linear {
            w: Var::new(),
            b: Var::new(),
        }
    }

    fn pass(&self, x: &Var) -> Var {
        w * x + b
    }
}

impl Net for Linear {
    fn params(&self) -> &[&Var] {
        &[&w, &b]
    }
}
