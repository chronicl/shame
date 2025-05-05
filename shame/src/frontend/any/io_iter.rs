use std::{cell::Cell, rc::Rc};

use crate::{
    any::{Attrib, Location, VertexBufferLayout},
    call_info,
    common::proc_macro_reexports::TypeLayoutSemantics,
    frontend::rust_types::error::FrontendError,
    ir::{self, pipeline::PipelineError, recording::Context},
    GpuLayout, TypeLayout, VertexIndex, VertexLayout,
};

use super::{Any, InvalidReason};

/// an iterator over the draw command's bound vertex buffers, which also
/// allows random access
///
/// use `.next()` or `.at(...)`/`.index(...)` to access individual vertex buffers
pub struct VertexBufferIterAny {
    next_slot: u32,
    private_ctor: (),
}

impl VertexBufferIterAny {
    pub(crate) fn new() -> Self {
        Self {
            next_slot: 0,
            private_ctor: (),
        }
    }

    /// access the `i`th vertex buffer
    /// that was bound when the draw command was scheduled and interpret its
    /// type as `gpu_layout`.
    #[track_caller]
    pub fn at(&mut self, i: u32, layout: PartialVertexBufferLayout) -> VertexBufferAny {
        self.next_slot = i + 1;
        VertexBufferAny::new(i, layout)
    }

    /// access the next vertex buffer (or the first if no buffer was imported yet)
    /// that was bound when the draw command was scheduled and interpret its
    /// type as `gpu_layout`.
    #[allow(clippy::should_implement_trait)] // not fallible
    #[track_caller]
    pub fn next(&mut self, layout: PartialVertexBufferLayout) -> VertexBufferAny {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.at(slot, layout)
    }
}

/// (no documentation - chronicl)
pub struct VertexBufferAny {
    slot: u32,
    layout: Result<PartialVertexBufferLayout, InvalidReason>,
    attribute_count: usize,
}

impl VertexBufferAny {
    #[track_caller]
    fn new(slot: u32, layout: PartialVertexBufferLayout) -> Self {
        let attribute_count = layout.attribs.len();

        let layout = Context::try_with(call_info!(), |ctx| {
            let rp = ctx.render_pipeline();
            match ensure_locations_are_unique(slot, ctx, &rp, &layout.attribs) {
                Err(e) => {
                    ctx.push_error(e.into());
                    Err(InvalidReason::ErrorThatWasPushed)
                }
                Ok(_) => Ok(layout),
            }
        })
        .unwrap_or(Err(InvalidReason::CreatedWithNoActiveEncoding));

        Self {
            slot,
            layout,
            attribute_count,
        }
    }

    /// (no documentation - chronicl)
    #[track_caller]
    pub fn index(self, index: VertexIndex) -> Vec<Any> {
        // just to be consistent with the other `at` functions, where the
        // `shame::Index` trait provides the `index` alternative, we offer both as
        // type associated functions here. Choose which one you like better.
        self.at(index)
    }

    /// (no documentation - chronicl)
    #[track_caller]
    pub fn at(self, index: VertexIndex) -> Vec<Any> {
        let lookup = index.0;

        let call_info = call_info!();
        let result = self.layout.and_then(|layout| {
            Context::try_with(call_info, |ctx| {
                Any::vertex_buffer(
                    self.slot,
                    VertexBufferLayout {
                        lookup,
                        stride: layout.stride,
                        attribs: layout.attribs,
                    },
                )
            })
            .ok_or(InvalidReason::CreatedWithNoActiveEncoding)
        });

        match result {
            Ok(anys) => anys,
            Err(reason) => vec![Any::new_invalid(reason); self.attribute_count],
        }
    }
}

/// (no documentation - chronicl)
pub struct PartialVertexBufferLayout {
    /// (no documentation - chronicl)
    pub attribs: Box<[Attrib]>,
    /// (no documentation - chronicl)
    pub stride: u64,
}

impl PartialVertexBufferLayout {
    #[track_caller]
    pub(crate) fn try_from_type<T: GpuLayout>(location_counter: &LocationCounter) -> Result<Self, FrontendError> {
        Self::try_from_type_layout(T::gpu_layout(), location_counter)
    }

    #[track_caller]
    pub(crate) fn try_from_type_layout(
        layout: TypeLayout,
        location_counter: &LocationCounter,
    ) -> Result<Self, FrontendError> {
        Attrib::get_attribs_and_stride(&layout, location_counter)
            .ok_or(FrontendError::MalformedVertexBufferLayout(layout))
            .map(|(attribs, stride)| Self { attribs, stride })
    }

    /// Infallible conversion support for types implementing `VertexLayout`.
    ///
    /// `VertexLayout` is automatically derived with `#[derive(GpuLayout)]` if the type can be converted
    /// to a vertex buffer layout. The requirements for that are
    /// TODO(chronicl) check what the exact requirements are
    /// - The type must not contain any nested types.
    /// - All fields of the type must implement VertexAttribute
    #[track_caller]
    pub(crate) fn from_vertex_layout<T: VertexLayout>(location_counter: &LocationCounter) -> Self {
        Self::try_from_type::<T>(location_counter).unwrap()
    }
}

/// (no documentation - chronicl)
pub struct LocationCounter(Cell<u32>);

impl LocationCounter {
    pub(crate) fn next(&self) -> Location {
        let i = self.0.get();
        self.0.set(i + 1);
        Location(i)
    }
}

impl From<u32> for LocationCounter {
    fn from(value: u32) -> Self { Self(Cell::new(value)) }
}

/// checks that there are no duplicate vertex attribute locations and vertex buffer slots
fn ensure_locations_are_unique(
    slot: u32,
    ctx: &Context,
    rp: &ir::pipeline::WipRenderPipelineDescriptor,
    new_attribs: &[Attrib],
) -> Result<(), PipelineError> {
    for vbuf in &rp.vertex_buffers {
        if vbuf.index == slot {
            return Err(PipelineError::DuplicateVertexBufferImport(slot));
        }
        for existing_attrib in &vbuf.attribs {
            if new_attribs.iter().any(|a| a.location == existing_attrib.location) {
                return Err(PipelineError::DuplicateAttribLocation {
                    location: existing_attrib.location,
                    buffer_a: vbuf.index,
                    buffer_b: slot,
                });
            }
        }
    }
    Ok(())
}
