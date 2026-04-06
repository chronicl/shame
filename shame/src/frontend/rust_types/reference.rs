use super::{
    array::{Array, ArrayLen},
    error::FrontendError,
    layout_traits::{FromAnys},
    mem::{self, AddressSpace},
    type_traits::{GpuSized, NoAtomics},
    typecheck_downcast,
    vec::ToInteger,
    AsAny, GpuType, To,
};
use crate::{
    GpuLayout, Len, ScalarTypeNumber, ToGpuType, call_info,
    frontend::{any::Any, rust_types::vec::vec},
    ir::{self, LayoutType, StoreType, Type, recording::Context},
    x1,
};
use std::borrow::Borrow;

/// A memory operation can read, write, or both read and write.
/// Memory locations may support only some of these accesses.
///
/// this trait is implemented by the following marker types
/// - [`Read`] for read-only access
/// - [`Write`] for write-only access
/// - [`ReadWrite`] for both read and write access
///
/// see https://www.w3.org/TR/WGSL/#memory-access-mode
pub trait AccessMode: Copy + 'static {
    #[doc(hidden)] // runtime api
    const ACCESS: ir::AccessMode;
}

/// read-only [`AccessMode`] marker type
#[derive(Clone, Copy)]
pub struct Read;
impl AccessMode for Read {
    const ACCESS: ir::AccessMode = ir::AccessMode::Read;
}

/// write-only [`AccessMode`] marker type
#[derive(Clone, Copy)]
pub struct Write;
impl AccessMode for Write {
    const ACCESS: ir::AccessMode = ir::AccessMode::Write;
}

/// read and write [`AccessMode`] marker type
#[derive(Clone, Copy)]
pub struct ReadWrite;
impl AccessMode for ReadWrite {
    const ACCESS: ir::AccessMode = ir::AccessMode::ReadWrite;
}

/// An [`AccessMode`] that is either [`Read`] or [`ReadWrite`]
pub trait AccessModeReadable: AccessMode {
    #[doc(hidden)] // runtime api
    const ACCESS_MODE_READABLE: ir::AccessModeReadable;
} // TODO(release) seal this trait
impl AccessModeReadable for ReadWrite {
    const ACCESS_MODE_READABLE: ir::AccessModeReadable = ir::AccessModeReadable::ReadWrite;
}
impl AccessModeReadable for Read {
    const ACCESS_MODE_READABLE: ir::AccessModeReadable = ir::AccessModeReadable::Read;
}

/// An [`AccessMode`] that is either [`Write`] or [`ReadWrite`]
pub trait AccessModeWritable: AccessMode {} // TODO(release) seal this trait
impl AccessModeWritable for ReadWrite {}
impl AccessModeWritable for Write {}

// TODO(docs) Docs: mention that this has broadly same interface as `Cell`
/// (no documentation yet)
pub struct Ref<T, AS = mem::Fn, AM = ReadWrite>
where
    T: GpuLayout,
    AS: AddressSpace,
    AM: AccessMode,
{
    any: Any,
    fields_as_refs: T::RefFields<AS, AM>,
}

impl<T: GpuLayout, AS: AddressSpace, AM: AccessMode> Copy for Ref<T, AS, AM> {}

impl<T: GpuLayout, AS: AddressSpace, AM: AccessMode> Clone for Ref<T, AS, AM> {
    fn clone(&self) -> Self { *self }
}

impl<T, AS, AM> AsAny for Ref<T, AS, AM>
where
    T: GpuLayout,
    AS: AddressSpace,
    AM: AccessMode,
{
    fn as_any(&self) -> Any { self.any }
}

// TODO: this is a misuse of ToGpuType
impl<T, AS, AM> ToGpuType for Ref<T, AS, AM>
where
    T: GpuSized + NoAtomics,
    AS: AddressSpace,
    AM: AccessModeReadable,
{
    type Gpu = T;
    fn to_gpu(&self) -> Self::Gpu { self.get() }
}

impl<T, AS, AM> Ref<T, AS, AM>
where
    T: GpuLayout + NoAtomics,
    AS: AddressSpace,
    AM: AccessModeReadable,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn get(&self) -> T
    where
        T: GpuSized, // value must be constructible
    {
        self.deref()
    }

    /// (no documentation yet)
    #[track_caller]
    pub fn deref(&self) -> T
    where
        T: GpuSized, // value must be constructible
    {
        self.any.ref_load().into()
    }
}

impl<T, AS, AM> std::ops::Deref for Ref<T, AS, AM>
where
    T: GpuLayout,
    AS: AddressSpace,
    AM: AccessMode,
{
    type Target = T::RefFields<AS, AM>;

    fn deref(&self) -> &Self::Target { &self.fields_as_refs }
}

impl<T, AS, AM> Ref<T, AS, AM>
where
    T: GpuType + GpuLayout + NoAtomics,
    AS: AddressSpace,
    AM: AccessModeWritable,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn set(&self, value: impl To<T>)
    where
        T: GpuSized, // value must be constructible
    {
        self.any.set(value.to_any())
    }
}

impl<T, AS, AM> From<Any> for Ref<T, AS, AM>
where
    T: GpuLayout,
    AS: AddressSpace,
    AM: AccessMode,
{
    #[track_caller]
    fn from(any: Any) -> Self { ref_from_any_and_layout_type(any, <T as GpuLayout>::layout_type_owned()) }
}

fn ref_from_any_and_layout_type<T, AS, AM>(any: Any, expected_store_ty: LayoutType) -> Ref<T, AS, AM>
where
    T: GpuLayout,
    AS: AddressSpace,
    AM: AccessMode,
{
    let from_any_unchecked = |any| Ref::<T, AS, AM> {
        any,
        fields_as_refs: {
            let field_anys = <T as GpuLayout>::fields_as_anys_unchecked(any);
            FromAnys::from_anys((field_anys.borrow() as &[Any]).iter().copied())
        },
    };
    Context::with(call_info!(), |ctx| {
        let invalid = |e| from_any_unchecked(ctx.push_error_get_invalid_any(e));

        match any.ty() {
            Some(Type::Ref(alloc, store_ty, access)) => match alloc.address_space == AS::ADDRESS_SPACE {
                true => typecheck_downcast(
                    any,
                    Type::Ref(alloc, expected_store_ty.into(), AM::ACCESS),
                    from_any_unchecked,
                ),
                false => invalid(
                    FrontendError::DowncastWithInvalidAddressSpace {
                        dynamic_as: alloc.address_space,
                        rust_as: AS::ADDRESS_SPACE,
                    }
                    .into(),
                ),
            },
            Some(ty) => invalid(
                FrontendError::DowncastNonRefToRef {
                    dynamic_type: ty,
                    rust_type: expected_store_ty.into(),
                }
                .into(),
            ),
            None => from_any_unchecked(any), // already invalid.
        }
    })
}

impl<T, AS, AM, N> Ref<Array<T, N>, AS, AM>
where
    T: GpuSized + 'static,
    AS: AddressSpace + 'static,
    AM: AccessMode + 'static,
    N: ArrayLen,
{
    /// (no documentation yet)
    #[track_caller]
    pub fn at(&self, index: impl ToInteger) -> Ref<T, AS, AM> { self.as_any().array_index(index.to_any()).into() }
}

impl<T: GpuSized + 'static, AS: AddressSpace, AM: AccessMode> Ref<Array<T>, AS, AM> {
    /// (no documentation yet)
    pub fn len(&self) -> vec<u32, x1> { self.any.address().array_length().into() }
}

impl<T, AM> Ref<T, mem::WorkGroup, AM>
where
    T: GpuLayout + NoAtomics,
    AM: AccessModeReadable,
{
    // see WGSL https://www.w3.org/TR/WGSL/#workgroupUniformLoad-builtin
    /// (no documentation yet)
    pub fn uniform_load(&self) -> T { self.as_any().address().workgroup_uniform_load().into() }
}

// Unary ops
impl<T, AS, AM> std::ops::Neg for Ref<T, AS, AM>
where
    T: GpuSized + NoAtomics,
    AS: AddressSpace,
    AM: AccessModeReadable,
    T: std::ops::Neg,
{
    type Output = <T as std::ops::Neg>::Output;
    fn neg(self) -> Self::Output { self.get().neg() }
}

impl<T, AS, AM> std::ops::Not for Ref<T, AS, AM>
where
    T: GpuSized + NoAtomics,
    AS: AddressSpace,
    AM: AccessModeReadable,
    T: std::ops::Not,
{
    type Output = <T as std::ops::Not>::Output;
    fn not(self) -> Self::Output { self.get().not() }
}

// Binary ops
macro_rules! impl_ref_binop {
    ($trait:ident, $method:ident) => {
        impl<T1, T2, AS, AM> std::ops::$trait<T1> for Ref<T2, AS, AM>
        where
            T2: GpuSized + NoAtomics,
            AS: AddressSpace,
            AM: AccessModeReadable,
            T2: std::ops::$trait<T1>,
        {
            type Output = <T2 as std::ops::$trait<T1>>::Output;
            fn $method(self, rhs: T1) -> Self::Output { self.get().$method(rhs) }
        }

        impl<N, L, T, AS, AM> std::ops::$trait<Ref<T, AS, AM>> for vec<N, L>
        where
            N: ScalarTypeNumber,
            L: Len,
            T: GpuSized + NoAtomics,
            AS: AddressSpace,
            AM: AccessModeReadable,
            vec<N, L>: std::ops::$trait<T>,
        {
            type Output = <vec<N, L> as std::ops::$trait<T>>::Output;
            fn $method(self, rhs: Ref<T, AS, AM>) -> Self::Output { self.$method(rhs.get()) }
        }

        impl<T, AS, AM> std::ops::$trait<Ref<T, AS, AM>> for u32
        where
            T: GpuSized + NoAtomics,
            AS: AddressSpace,
            AM: AccessModeReadable,
            u32: std::ops::$trait<T>,
        {
            type Output = <u32 as std::ops::$trait<T>>::Output;
            fn $method(self, rhs: Ref<T, AS, AM>) -> Self::Output { self.$method(rhs.get()) }
        }

        impl<T, AS, AM> std::ops::$trait<Ref<T, AS, AM>> for i32
        where
            T: GpuSized + NoAtomics,
            AS: AddressSpace,
            AM: AccessModeReadable,
            i32: std::ops::$trait<T>,
        {
            type Output = <i32 as std::ops::$trait<T>>::Output;
            fn $method(self, rhs: Ref<T, AS, AM>) -> Self::Output { self.$method(rhs.get()) }
        }

        impl<T, AS, AM> std::ops::$trait<Ref<T, AS, AM>> for f32
        where
            T: GpuSized + NoAtomics,
            AS: AddressSpace,
            AM: AccessModeReadable,
            f32: std::ops::$trait<T>,
        {
            type Output = <f32 as std::ops::$trait<T>>::Output;
            fn $method(self, rhs: Ref<T, AS, AM>) -> Self::Output { self.$method(rhs.get()) }
        }

        // impl<S, T, AS, AM> std::ops::$trait<Ref<T, AS, AM>> for crate::Struct<S>
        // where
        //     S: SizedFields,
        //     T: GpuSized + NoAtomics,
        //     AS: AddressSpace,
        //     AM: AccessModeReadable,
        //     crate::Struct<S>: std::ops::$trait<T>,
        // {
        //     type Output = <crate::Struct<S> as std::ops::$trait<T>>::Output;
        //     fn $method(self, rhs: Ref<T, AS, AM>) -> Self::Output { self.$method(rhs.get()) }
        // }

        // This probably doesn't really implement much
        impl<A, T, AS, AM, const N: usize> std::ops::$trait<Ref<T, AS, AM>> for crate::Array<A, crate::Size<N>>
        where
            T: GpuSized + NoAtomics,
            AS: AddressSpace,
            AM: AccessModeReadable,
            crate::Array<A, crate::Size<N>>: std::ops::$trait<T>,
        {
            type Output = <crate::Array<A, crate::Size<N>> as std::ops::$trait<T>>::Output;
            fn $method(self, rhs: Ref<T, AS, AM>) -> Self::Output { self.$method(rhs.get()) }
        }
    };
}

impl_ref_binop!(Add, add);
impl_ref_binop!(Sub, sub);
impl_ref_binop!(Mul, mul);
impl_ref_binop!(Div, div);
impl_ref_binop!(Rem, rem);
impl_ref_binop!(BitAnd, bitand);
impl_ref_binop!(BitOr, bitor);
impl_ref_binop!(BitXor, bitxor);
impl_ref_binop!(Shl, shl);
impl_ref_binop!(Shr, shr);

#[test]
fn test_ref_ops() {
    let mut enc = crate::start_encoding(Default::default()).unwrap();
    enc.new_render_pipeline(Default::default());
    macro_rules! test_ops {
        ($a:expr, $ax2:expr) => {
            let a = $a;
            let ax2 = $ax2;
            // a_raw + ax2_raw; // fails
            a + a;
            ax2 + ax2;
            a + 1;
            ax2 + 1;
            1 + a;
            1 + ax2;
            1 | a;
            a | 1;
            ax2 << 1;
            ax2 << a;
        };
    }

    test_ops!(0u32.to_gpu(), crate::vec![1u32, 2u32]);
    test_ops!(crate::Cell::new(0u32), crate::Cell::new(crate::vec![1u32, 2u32]));
}
