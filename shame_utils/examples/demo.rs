use shame::aliases::*;
use shame_utils::{Layout, NoPadding};

#[derive(NoPadding, Clone, Copy)]
#[rustfmt::skip]
struct A {
    a: f32x1,  _0: f32x1, _1: f32x1, _2: f32x1, // or _0: Array<f32x1, 3> if not used in uniform buffer
    b: f32x3,  _3: f32x1,
}

fn main() {}
