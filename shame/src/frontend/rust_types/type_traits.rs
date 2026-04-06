use std::borrow::Borrow;

use super::{
    layout_traits::{FromAnys},
    mem::AddressSpace,
    reference::AccessMode,
    AsAny, GpuType,
};
use crate::{GpuLayout, frontend::any::shared_io::BindPath};
use crate::{
    call_info,
    frontend::{
        any::{render_io::VertexAttribFormat, Any},
        error::InternalError,
    },
    ir::{
        self,
        ir_type::{AlignedType},
        pipeline::StageMask,
        recording::Context,
    },
};

/// marker type for `impl Store` to specify that `Self` has no particular `Store::RefFields`
#[derive(Clone, Copy)]
pub struct EmptyRefFields;

impl FromAnys for EmptyRefFields {
    fn expected_num_anys() -> usize { 0 }

    #[track_caller]
    fn from_anys(anys: impl Iterator<Item = Any>) -> Self {
        match anys.count() {
            0 => (),
            n => {
                let err = format!(
                    "trying to instantiate {} (which takes 0 arguments) with {n} argument.",
                    std::stringify!(EmptyRefFields)
                );
                Context::try_with(call_info!(), |ctx| {
                    ctx.push_error_get_invalid_any(InternalError::new(true, err).into())
                });
            }
        };
        Self
    }
}

#[doc(hidden)]
pub struct BindingArgs {
    pub path: BindPath,
    pub visibility: StageMask,
}

//[old-doc] A type whose values can be stored in a shader memory cell
//[old-doc] (for example via buffer bindings or by allocating via [`shame::alloc`])
//[old-doc] and manipulated/accessed via
//[old-doc] a [`shame::Ref`].
//[old-doc]
//[old-doc] corresponds to WGSL "Storable type" https://www.w3.org/TR/WGSL/#storable-types
/// (no documentation yet)
pub trait GpuStore: GpuType {
    #[doc(hidden)] // runtime api
    fn store_ty() -> ir::StoreType;

    #[doc(hidden)] // unstable
    #[track_caller]
    /// forces `self` to appear in the generated shader code. No dead code elimination can remove it.
    fn show(&self) -> &Self {
        self.as_any().show();
        self
    }
}

#[diagnostic::on_unimplemented(message = "the size of `{Self}` on the gpu is not known at rust compile-time")]
/// ## known byte-size on the gpu
/// types whose byte-size on the graphics device is known at rust compile-time
///
/// This is also implemented for non-[`GpuType`]s like structs which derive [`GpuLayout`]
/// which contain only fields that are [`GpuSized`]
///
/// note: [`GpuSized`] does not imply [`GpuStore`], because [`Atomic<T>`] is [`GpuSized`] but `!GpuStore`
///
/// [`Atomic<T>`]: crate::Atomic
pub trait GpuSized: GpuLayout {
    /// The sized layout of `Self` on the gpu.
    const LAYOUT_SIZED: crate::layout::SizedType<'static>;
}

/// this trait is only implemented by:
///
/// * `sm::vec`s of non-boolean type (e.g. `sm::f32x4`)
/// * `sm::packed::PackedVec`s (e.g. `sm::packed::unorm8x4`)
// Is at most 16 bytes according to https://www.w3.org/TR/WGSL/#input-output-locations
// and thus GpuSized.
pub trait VertexAttribute: FromAnys {
    #[doc(hidden)] // runtime api
    fn vertex_attrib_format() -> VertexAttribFormat;
}

/// Trait that the fields of a derived `GpuLayout` type must implement.
/// This is used for showing a more helpful error message when trying to use
/// #[derive(GpuLayout)]
/// struct A { ... }
/// in
/// #[derive(GpuLayout)]
/// struct B { a: A }
/// directly. This should instead be
/// #[derive(GpuLayout)]
/// struct B { a: shame::Struct<A> }
/// which the error message points out.
#[diagnostic::on_unimplemented(
    message = "{Self} is not a valid `shame::GpuLayout` field type. These include `shame::GpuType`s and `shame::packed::PackedVec`. If {Self} is a `shame::GpuLayout` struct, it can be used as a field by wrapping it in `shame::Struct<{Self}>`."
)]
pub trait GpuLayoutField {
    /// Constructs Self from Any
    fn from_any(any: Any) -> Self;
}

impl<T: From<Any>> GpuLayoutField for T {
    fn from_any(any: Any) -> Self { T::from(any) }
}
