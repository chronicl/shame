use std::{cell::Cell, rc::Rc};

use crate::{
    any::{Attrib, Location, VertexBufferLayout},
    call_info,
    common::proc_macro_reexports::TypeLayoutSemantics,
    frontend::rust_types::error::FrontendError,
    ir::{self, pipeline::PipelineError, recording::Context},
    TypeLayout, VertexIndex,
};

use super::{Any, InvalidReason};

/// an iterator over the draw command's bound vertex buffers, which also
/// allows random access
///
/// use `.next()` or `.at(...)`/`.index(...)` to access individual vertex buffers
pub struct VertexBufferIterAny {
    next_slot: u32,
    location_counter: Rc<LocationCounter>,
    private_ctor: (),
}

impl VertexBufferIterAny {
    pub(crate) fn new() -> Self {
        Self {
            next_slot: 0,
            location_counter: Rc::new(0.into()),
            private_ctor: (),
        }
    }

    /// access the `i`th vertex buffer
    /// that was bound when the draw command was scheduled and interpret its
    /// type as `gpu_layout`.
    #[track_caller]
    pub fn at(&mut self, i: u32, gpu_layout: TypeLayout) -> VertexBufferAny {
        self.next_slot = i + 1;
        VertexBufferAny::new(i, self.location_counter.clone(), gpu_layout)
    }

    /// access the next vertex buffer (or the first if no buffer was imported yet)
    /// that was bound when the draw command was scheduled and interpret its
    /// type as `gpu_layout`.
    #[allow(clippy::should_implement_trait)] // not fallible
    #[track_caller]
    pub fn next(&mut self, gpu_layout: TypeLayout) -> VertexBufferAny {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.at(slot, gpu_layout)
    }
}

pub struct VertexBufferAny {
    slot: u32,
    attribs_and_stride: Result<(Box<[Attrib]>, u64), InvalidReason>,
    expected_any_count: usize,
}

impl VertexBufferAny {
    #[track_caller]
    fn new(slot: u32, location_counter: Rc<LocationCounter>, gpu_layout: TypeLayout) -> Self {
        // Is this correct?
        let expected_any_count = match &gpu_layout.kind {
            TypeLayoutSemantics::Structure(s) => s.fields.len(),
            _ => 1,
        };

        let call_info = call_info!();
        let attribs_and_stride = Context::try_with(call_info, |ctx| {
            let attribs_and_stride = Attrib::get_attribs_and_stride(&gpu_layout, &location_counter).ok_or_else(|| {
                ctx.push_error(FrontendError::MalformedVertexBufferLayout(gpu_layout).into());
                InvalidReason::ErrorThatWasPushed
            });

            if let Ok((new_attribs, _)) = &attribs_and_stride {
                let rp = ctx.render_pipeline();
                if let Err(e) = ensure_locations_are_unique(slot, ctx, &rp, new_attribs) {
                    ctx.push_error(e.into());
                }
            }

            attribs_and_stride
        })
        .unwrap_or(Err(InvalidReason::CreatedWithNoActiveEncoding));

        Self {
            slot,
            attribs_and_stride,
            expected_any_count,
        }
    }

    #[track_caller]
    pub fn index(self, index: VertexIndex) -> Vec<Any> {
        // just to be consistent with the other `at` functions, where the
        // `shame::Index` trait provides the `index` alternative, we offer both as
        // type associated functions here. Choose which one you like better.
        self.at(index)
    }

    #[track_caller]
    pub fn at(self, index: VertexIndex) -> Vec<Any> {
        let lookup = index.0;

        let result = Context::try_with(call_info!(), |ctx| {
            self.attribs_and_stride.map(|(attribs, stride)| {
                Any::vertex_buffer(
                    self.slot,
                    VertexBufferLayout {
                        stride,
                        lookup,
                        attribs,
                    },
                )
            })
        })
        .unwrap_or(Err(InvalidReason::CreatedWithNoActiveEncoding));

        match result {
            Ok(anys) => anys,
            Err(reason) => vec![Any::new_invalid(reason); self.expected_any_count],
        }
    }
}

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
