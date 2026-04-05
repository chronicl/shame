#![allow(dead_code, unused)]
use shame::{aliases::*, Array, GpuLayout, Size};
use shame_utils::{Layout, NoPadding, ToGlam, gpu_code};
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
    b: B,
}

#[derive(GpuLayout, NoPadding, ToGlam, Clone, Copy)]
#[rustfmt::skip]
struct B {
    b: Array<f32x3,Size<3>>,
    a: f32x3,                 _0: f32x1,
}

fn gpu_code_macro_example() {
    gpu_code! {
        let mut a = 0u32;
        let mut b = [0u32; 10];
        for i in 0..10u32 {
            if i % 2 == 0u32 {
                a += a * 2 + 1;
                b[i] = a;
            } else {
                a += 1u32;
            }
        }
    }

    // expands to
    let a = ::shame::Cell::new(0u32);
    let b = ::shame::Cell::new([0u32; 10]);
    ::shame::for_range(0..10u32, |i| {
        ::shame::if_else(
            (i % 2).equals((0u32)),
            || {
                (a).set_add((a * 2 + 1));
                ((b).at((i))).set((a));
            },
            || {
                (a).set_add((1u32));
            },
        );
    });
}
