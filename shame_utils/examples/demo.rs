#![allow(dead_code, unused)]
use shame::{aliases::*, Array, GpuLayout, Size, Struct};
use shame_utils::{Layout, NoPadding, ToGlam};
use bytemuck::{Pod, Zeroable};


#[derive(GpuLayout, NoPadding, ToGlam, Clone, Copy)]
#[cpu_derive(Default, Pod, Zeroable, Debug, Clone, Copy)]
#[rustfmt::skip]
struct Vertex {
    uv:  f32x2,  _0: f32x2,
    pos: f32x3,  _1: f32x1,
    nor: f32x3,  _2: f32x1,
}

fn main() {
    let vertex = VertexCpu {
        uv: glam::Vec2::ZERO,
        pos: glam::Vec3::ZERO,
        nor: glam::Vec3::ZERO,
        ..Default::default()
    };
    let vertex = VertexCpu::zeroed();

    let s = UnsizedCpu {
        a: glam::Vec3::ZERO,
        b: vec![glam::Vec2::ZERO, glam::Vec2::ZERO],
        ..Default::default()
    };
}

#[derive(GpuLayout, NoPadding, ToGlam, Clone)]
#[cpu_derive(Default)]
#[rustfmt::skip]
struct Unsized {
    a: f32x3,         _0: f32x1,
    b: Array<f32x2>,
}

#[derive(GpuLayout, NoPadding, ToGlam, Clone)]
#[rustfmt::skip]
struct A {
    a: f32x3,      _0: f32x1,
    b: Struct<B>,
}

#[derive(GpuLayout, NoPadding, ToGlam, Clone, Copy)]
#[rustfmt::skip]
struct B {
    b: Array<f32x3,Size<3>>,
    a: f32x3,                 _0: f32x1,
}
