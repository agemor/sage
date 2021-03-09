use crate::autodiff::Var;

pub trait ToVar {
    fn to_var(&self) -> Var;
}
