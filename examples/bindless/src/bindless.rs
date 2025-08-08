#![allow(clippy::collapsible_match)]

use std::f32::consts::TAU;
use shame::mem::Uniform;
use shame::pipeline_kind::Render;
use shame::tf::Rgba8Unorm;
use shame::{BindGroupIter, BindingArray, Buffer, CpuLayout, DrawContext, EncodingGuard, GpuLayout, MipFn, Texture};
use shame_wgpu::bind_group::AsBindGroupLayout;
use thiserror::Error;

use wgpu::util::{BufferInitDescriptor, DeviceExt as _};
use wgpu::{BufferDescriptor, LoadOp::*, StoreOp::*};

use shame_wgpu::{self as sm, PipelineEncoder};
use sm::texture_view::TextureViewExt;
use sm::aliases::*;
use sm::prelude::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ShameWgpu(#[from] shame_wgpu::Error),
}

#[derive(CpuLayout, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct TextureGridCpu {
    texture_count: [u32; 2],
    spacing: f32,
    padding: f32,
}

#[derive(GpuLayout)]
struct TextureGrid {
    texture_count: u32x2,
    // spacing between and around textures
    // percentage of screen size in range [0.0, 1.0]
    spacing: f32x1,
    padding: f32x1,
}

impl TextureGrid {
    fn texture_size(&self) -> f32x1 {
        ((1.0 - (self.texture_count + 1).to_f32() * self.spacing) / self.texture_count.to_f32()).max_comp()
    }

    fn texture_index_from_uv(&self, coords: f32x2) -> (boolx1, u32x1, f32x2) {
        let spacing_plus_tex = self.spacing + self.texture_size();
        let spacing_percentage = (self.spacing.to_f32() / spacing_plus_tex).splat();
        let spacing_plus_tex_index = coords / spacing_plus_tex;
        let percentage = spacing_plus_tex_index.dfloor();
        let index = spacing_plus_tex_index.floor().to_u32();

        let is_in_bounds = index.lt(self.texture_count).all();
        let not_in_spacing = percentage.ge(spacing_percentage).all();

        let uv = (percentage - spacing_percentage) / (1.0 - spacing_percentage);

        let index = index.x + index.y * self.texture_count.x;

        (is_in_bounds & not_in_spacing, index, uv)
    }
}

struct Fullscreen<'a> {
    bind_groups: BindGroupIter<'a>,
    frag: sm::FragmentStage,
    uv: f32x2,
}

fn fullscreen_vertex_shader(enc: &mut EncodingGuard<Render>) -> Fullscreen<'_> {
    let mut drawcall = enc.new_render_pipeline(sm::Indexing::Incremental);

    let vertex_index = drawcall.vertices.index.to_u32();

    let uv = sm::vec![(vertex_index >> 1u32).to_f32(), (vertex_index & 1u32).to_f32()] * 2.0;
    let clip_position = sm::vec![uv.mul_each(sm::vec![2.0, -2.0]) + sm::vec![-1.0, 1.0], 0.0, 1.0];

    let frag = drawcall
        .vertices
        .assemble(clip_position, sm::Draw::triangle_list(sm::Winding::Ccw))
        .rasterize(sm::Accuracy::default());

    let uv = frag.fill_rate(sm::Fill::Linear, uv);

    let bind_groups = drawcall.bind_groups;
    Fullscreen { bind_groups, frag, uv }
}

pub struct BindlessExample {
    pipeline: wgpu::RenderPipeline,
    texture_count: (u32, u32),
    resources: BindlessResources,
}

struct BindlessResources {
    bind_group: wgpu::BindGroup,

    sampler: wgpu::Sampler,
    textures: Vec<wgpu::Texture>,
    texture_views: Vec<wgpu::TextureView>,
    texture_grid: wgpu::Buffer,
    tex_indices: Vec<wgpu::Buffer>,
}

shame_wgpu::bind_group! {
    struct Bindless2 {
        texture_grid: Buffer<TextureGrid>,
        sampler: sm::Sampler<sm::Nearest>,
        textures: BindingArray<Texture<Rgba8Unorm>>,
        tex_indices: BindingArray<Buffer<u32x1>>,
    }
}

impl BindlessExample {
    pub fn new(gpu: &sm::Gpu, texture_count: (u32, u32)) -> Result<Self, Error> {
        let pipeline = Self::new_pipeline(gpu)?;
        let resources = Self::new_resources(gpu, &pipeline, texture_count)?;
        Ok(Self {
            pipeline,
            texture_count,
            resources,
        })
    }

    fn new_pipeline(gpu: &sm::Gpu) -> Result<wgpu::RenderPipeline, Error> {
        let mut enc = gpu.create_pipeline_encoder(Default::default())?;
        let mut fullscreen = fullscreen_vertex_shader(&mut enc.enc_guard);

        let mut group = fullscreen.bind_groups.next();
        let bindings = Bindless2::from(group);

        let (is_tex, tex_index, tex_uv) = bindings.texture_grid.texture_index_from_uv(fullscreen.uv);

        // If the fragment is within the bounds of a texture, sample it.
        // The texture index is remapped by tex_indices.
        let mut color = sm::Cell::new(sm::vec![0.0, 0.0, 0.0, 1.0]);
        sm::if_(is_tex, move || {
            let tex_index = *bindings.tex_indices.at(tex_index);
            color.set(
                bindings
                    .sampler
                    .sample(bindings.textures.at(tex_index), MipFn::zero(), tex_uv),
            );
        });

        fullscreen
            .frag
            .attachments
            .color_iter()
            .next::<sm::SurfaceFormat>()
            .blend(sm::Blend::add(), color.get());

        Ok(enc.finish()?)
    }

    fn new_resources(
        gpu: &sm::Gpu,
        pipeline: &wgpu::RenderPipeline,
        texture_count: (u32, u32),
    ) -> Result<BindlessResources, Error> {
        let texture_grid = gpu.create_buffer_init(&BufferInitDescriptor {
            label: "texture_grid".into(),
            contents: bytemuck::bytes_of(&TextureGridCpu {
                texture_count: [texture_count.0, texture_count.1],
                spacing: 0.05,
                padding: 0.,
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let sampler = gpu.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bindless_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create textures with different colors
        let texture_count = texture_count.0 * texture_count.1;
        let texture_size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };

        let mut textures = Vec::new();
        let mut texture_views = Vec::new();
        for i in 0..texture_count {
            let color = color_gradient(i, texture_count);

            let texture_data: Vec<u8> = (0..texture_size.width * texture_size.height)
                .flat_map(|_| color.iter().copied())
                .collect();

            let texture = gpu.create_texture_with_data(
                gpu.queue(),
                &wgpu::TextureDescriptor {
                    label: Some(&format!("bindless_texture_{}", i)),
                    size: texture_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &texture_data,
            );

            let texture_view = texture.create_view(&Default::default());
            textures.push(texture);
            texture_views.push(texture_view);
        }

        // Create texture indices buffer
        use rand::seq::SliceRandom;
        let mut tex_indices_data: Vec<u32> = (0..texture_count).collect();
        tex_indices_data.shuffle(&mut rand::rng());
        let mut tex_indices = Vec::new();
        for i in tex_indices_data {
            tex_indices.push(gpu.create_buffer_init(&BufferInitDescriptor {
                label: "texture_indices".into(),
                contents: bytemuck::cast_slice(std::slice::from_ref(&i)),
                usage: wgpu::BufferUsages::STORAGE,
            }));
        }

        // Create bind group
        let bind_group_layout = Bindless2::create_bind_group_layout(gpu).unwrap();
        let bind_group = bind_group_layout.new_bind_group(
            gpu,
            Bindless2Resources {
                texture_grid: &texture_grid.as_entire_buffer_binding(),
                sampler: &sampler,
                textures: &texture_views.iter().collect::<Vec<_>>(),
                tex_indices: &tex_indices
                    .iter()
                    .map(|b| b.as_entire_buffer_binding())
                    .collect::<Vec<_>>(),
            },
        );

        Ok(BindlessResources {
            bind_group,
            sampler,
            texture_grid,
            textures,
            texture_views,
            tex_indices,
        })
    }

    pub fn submit_render_commands_to_gpu(&mut self, gpu: &sm::Gpu, surface: &wgpu::TextureView) -> Result<(), Error> {
        let mut cmd = gpu.create_command_encoder(&Default::default());
        {
            let mut pass = cmd.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[surface.attach_as_color(Clear(wgpu::Color::BLACK), Store)],
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.resources.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        let _ticket = gpu.queue().submit([cmd.finish()]);

        gpu.poll(wgpu::PollType::Poll);
        Ok(())
    }

    pub fn window_event(&mut self, event: &winit::event::WindowEvent) -> Result<(), Error> { Ok(()) }
}

fn color_gradient(i: u32, max: u32) -> [u8; 4] {
    // Create a smooth color transition based on i
    let t = i as f32 / (max as f32).max(1.0);
    let hue = t * TAU;
    // Full rotation through color space

    // Convert HSV to RGB (with S=1, V=1 for vibrant colors)
    let c = 1.0;
    let x = c * (1.0 - ((hue / (TAU / 6.0)) % 2.0 - 1.0).abs());
    let m = 0.0;

    let (r, g, b) = if hue < TAU / 6.0 {
        (c, x, 0.0)
    } else if hue < 2.0 * TAU / 6.0 {
        (x, c, 0.0)
    } else if hue < 3.0 * TAU / 6.0 {
        (0.0, c, x)
    } else if hue < 4.0 * TAU / 6.0 {
        (0.0, x, c)
    } else if hue < 5.0 * TAU / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let color = [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
        255u8,
    ];
    color
}
