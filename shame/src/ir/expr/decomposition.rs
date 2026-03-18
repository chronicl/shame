use std::{borrow::Cow, fmt::Display, num::NonZeroU32, ops::Deref, rc::Rc};

use crate::frontend::any::Any;
use crate::TextureSampleUsageType;
use crate::ir::StructKindRef;
use crate::ir::expr::type_check;
use crate::ir::{LayoutType, Vector};
use crate::{
    call_info,
    common::small_vec::SmallVec,
    frontend::any::record_node,
    impl_track_caller_fn_any,
    ir::{
        self,
        expr::type_check::{SigFormatting, SignatureStrings},
        CanonName, Len, Len2, ScalarType, SizedType, StoreType,
        recording::CallInfo,
        Type,
    },
    sig,
};
use std::fmt::Write;

use super::{Expr, NoMatchingSignature, TypeCheck};

/// one of the four vector components x, y, z, w
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Comp4 {
    X,
    Y,
    Z,
    W,
}

impl Comp4 {
    /// the lowercase letter of this component
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Comp4::X => "x",
            Comp4::Y => "y",
            Comp4::Z => "z",
            Comp4::W => "w",
        }
    }
}

impl std::fmt::Display for Comp4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.as_str()) }
}

impl Comp4 {
    /// the numeric index of the component, starting at 0 for x
    pub(crate) fn as_index(&self) -> u32 {
        match self {
            Comp4::X => 0,
            Comp4::Y => 1,
            Comp4::Z => 2,
            Comp4::W => 3,
        }
    }

    pub(crate) fn is_contained_in(&self, len: Len) -> bool { self.as_index() < u32::from(len) }
}

#[doc(hidden)] // runtime api
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorAccess {
    Swizzle1([Comp4; 1]),
    Swizzle2([Comp4; 2]),
    Swizzle3([Comp4; 3]),
    Swizzle4([Comp4; 4]),
}

impl VectorAccess {
    fn has_duplicate_components(&self) -> bool { (0..self.len()).any(|i| self[i + 1..].contains(&self[i])) }

    fn get_len(&self) -> Len {
        match self {
            Self::Swizzle1(..) => Len::X1,
            Self::Swizzle2(..) => Len::X2,
            Self::Swizzle3(..) => Len::X3,
            Self::Swizzle4(..) => Len::X4,
        }
    }
}

impl std::ops::Deref for VectorAccess {
    type Target = [Comp4];
    fn deref(&self) -> &Self::Target {
        match self {
            VectorAccess::Swizzle1(x) => x,
            VectorAccess::Swizzle2(x) => x,
            VectorAccess::Swizzle3(x) => x,
            VectorAccess::Swizzle4(x) => x,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decomposition {
    VectorAccess(VectorAccess),
    VectorIndex,
    VectorIndexConst(u32),

    MatrixIndex,
    // (no test case yet)
    MatrixIndexConst(u32),

    ArrayIndex,
    // (no test case yet)
    ArrayIndexConst(u32),

    // (no test case yet)
    BindingArrayIndex,
    // (no test case yet)
    BindingArrayIndexConst(u32),

    StructureAccess(CanonName),
}

impl Display for Decomposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Decomposition::VectorAccess(swizzle) => {
                write!(f, "vec.")?;
                for comp in swizzle.deref() {
                    f.write_str(comp.as_str())?;
                }
                Ok(())
            }
            Decomposition::VectorIndex => write!(f, "vec[_]"),
            Decomposition::VectorIndexConst(i) => write!(f, "vec[const {}]", i),
            Decomposition::MatrixIndex => write!(f, "mat[_]"),
            Decomposition::MatrixIndexConst(i) => write!(f, "mat[const {}]", i),
            Decomposition::ArrayIndex => write!(f, "array[_]"),
            Decomposition::ArrayIndexConst(i) => write!(f, "array[const {}]", i),
            Decomposition::BindingArrayIndex => write!(f, "binding-array[_]"),
            Decomposition::BindingArrayIndexConst(i) => write!(f, "binding-array[const {}]", i),
            Decomposition::StructureAccess(canon_name) => write!(f, "struct.{}", canon_name),
        }
    }
}

impl TypeCheck for Decomposition {
    #[rustfmt::skip]
    fn infer_type(&self, args: &[Type]) -> Result<Type, NoMatchingSignature> {
        use Len::*;
        use ScalarType::*;
        use StoreType::*;
        use Type::*;
        use type_check::{vec_store, mat_store, array_store};

        match self {
            Decomposition::VectorAccess(comps) => sig!(
                {
                    name: Decomposition::VectorAccess(comps),
                    comment_below: "where `comps` is a list of up to 4 components (x, y, z or w).",
                    fmt: SigFormatting::RemoveAsterisksAndClone,
                },
                [Store(     vec_store!(l, t)    )] if comps.len() <= *l => Store(Vector::new(*t, comps.get_len()).into()),
                [Ref(alloc, vec_store!(l, t), am)] if comps.len() == 1 => Ref(alloc.clone(), Vector::new(*t, comps.get_len()).into(), *am),
            )(self, args),
            Decomposition::VectorIndex => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Store(     vec_store!(l, t)    ), Store(vec_store!(X1, U32 | I32))] => Store(Vector::new(*t, X1).into()),
                [Ref(alloc, vec_store!(l, t), am), Store(vec_store!(X1, U32 | I32))] => Ref(alloc.clone(), Vector::new(*t, X1).into(), *am),
            )(self, args),
            Decomposition::VectorIndexConst(i) => sig!(
                {
                    name: Decomposition::VectorAccess(i),
                    fmt: SigFormatting::RemoveAsterisksAndClone,
                },
                [Store(     vec_store!(l, t)    )] if *i < *l => Store(Vector::new(*t, X1).into()),
                [Ref(alloc, vec_store!(l, t), am)] if *i < *l => Ref(alloc.clone(), Vector::new(*t, X1).into(), *am),
            )(self, args),
            Decomposition::MatrixIndex => sig!(
                [Store(     mat_store!(col, row, t)    ), Store(vec_store!(X1, U32 | I32))] => Store(Vector::new((*t).into(), (*row).into()).into()),
                [Ref(alloc, mat_store!(col, row, t), am), Store(vec_store!(X1, U32 | I32))] => Ref(alloc.clone(), Vector::new((*t).into(), (*row).into()).into(), *am),
            )(self, args),
            Decomposition::MatrixIndexConst(i) => sig!(
                {
                    name: Decomposition::MatrixIndexConst(i),
                    fmt: SigFormatting::RemoveAsterisksAndClone,
                },
                [Store(     mat_store!(col, row, t)    )] if *i < *col => Store(Vector::new((*t).into(), (*row).into()).into()),
                [Ref(alloc, mat_store!(col, row, t), am)] if *i < *col => Ref(alloc.clone(), Vector::new((*t).into(), (*row).into()).into(), *am),
            )(self, args),
            Decomposition::ArrayIndex => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                [Store(     array_store!(t, _)), Store(vec_store!(X1, I32 | U32))] => Type::from((**t).clone()),
                [Ref(alloc, array_store!(t, _), am), Store(vec_store!(X1, I32 | U32))] => Ref(alloc.clone(), (**t).clone().into(), *am),
                [Ref(alloc, StoreType::Layout(LayoutType::RuntimeSizedArray(t)), am), Store(vec_store!(X1, I32 | U32))] => Ref(alloc.clone(), t.element.clone().into(), *am),
            )(self, args),
            Decomposition::ArrayIndexConst(i) => sig!(
                {
                    name: Decomposition::ArrayIndexConst(i),
                    fmt: SigFormatting::RemoveAsterisksAndClone,
                },
                [Store(     array_store!(t, n))] if *i < n.get() => Type::from((**t).clone()),
                [Ref(alloc, array_store!(t, n), am)] if *i < n.get() => Ref(alloc.clone(), (**t).clone().into(), *am),
                [Ref(alloc, StoreType::Layout(LayoutType::RuntimeSizedArray(t)), am)] => Ref(alloc.clone(), t.element.clone().into(), *am),
            )(self, args),
            Decomposition::BindingArrayIndex => sig!(
                { fmt: SigFormatting::RemoveAsterisksAndClone, },
                // For texture binding arrays
                [Store(BindingArray(t, n)), Store(vec_store!(X1, I32 | U32))] => Type::Store((**t).clone()),
                // For buffer binding arrays
                [Ref(alloc, BindingArray(t, n), am), Store(vec_store!(X1, I32 | U32))] => Ref(alloc.clone(), (**t).clone(), *am),
            )(self, args),
            Decomposition::BindingArrayIndexConst(i) => sig!(
                {
                    name: Decomposition::BindingArrayIndexConst(i),
                    fmt: SigFormatting::RemoveAsterisksAndClone,
                },
                [Store(BindingArray(t, n))] if n.map(|n| *i < n.get()).unwrap_or(true) => Type::Store((**t).clone()),
                [Ref(alloc, BindingArray(t, n), am)] if n.map(|n| *i < n.get()).unwrap_or(true) => Ref(alloc.clone(), (**t).clone(), *am),
            )(self, args),
            Decomposition::StructureAccess(field) => self.infer_type_of_structure_access(args, field),
        }
    }
}

impl Decomposition {
    #[rustfmt::skip]
    fn infer_type_of_structure_access(
        &self,
        args: &[Type],
        field_name: &CanonName,
    ) -> Result<Type, NoMatchingSignature> {
        use SizedType::*;
        use StoreType::*;
        use Type::*;

        let no_matching_sig_with_comment = |comment| NoMatchingSignature {
            expression_name: format!("{self:?}").into(),
            arguments: args.iter().cloned().collect(),
            allowed_signatures: SignatureStrings::Static(&[]),
            shorthand_level: Default::default(),
            signature_formatting: None,
            comment: Some(comment),
        };

        let single_arg = match args {
            [arg] => arg,
            _ => {
                return Err(no_matching_sig_with_comment(
                    "struct access expressions require
            having only a single (self) argument."
                        .into(),
                ))
            }
        };

        let struct_: StructKindRef<'_> = match single_arg {
            Ref(_, StoreType::Layout(LayoutType::Sized(SizedType::Struct(s))), _) |
            Store(StoreType::Layout(LayoutType::Sized(SizedType::Struct(s)))) => s.into(),

            Ref(_, StoreType::Layout(LayoutType::UnsizedStruct(s)), _) => s.into(),

            _ => {
                return Err(no_matching_sig_with_comment(
                    "struct access expressions require
                a single (self) argument of type struct/buffer-block or \
                reference-to-struct/buffer-block."
                        .into(),
                ))
            }
        };

        match struct_.find_field(field_name) {
            Some(field) => Ok(match single_arg {
                Ref(alloc, _, am) => Type::Ref(alloc.clone(), StoreType::Layout(field), *am),
                Ptr(alloc, _, am) => Type::Ref(alloc.clone(), StoreType::Layout(field), *am),
                Store(_) | Unit => Type::Store(StoreType::Layout(field)),
            }),
            None => Err(no_matching_sig_with_comment({
                let mut s = String::new();
                writeln!(s, "struct `{}` has no field named `{}`.", struct_.name(), field_name);

                if struct_.is_empty() {
                    writeln!(s, "this struct has no fields.");
                } else {
                    writeln!(s, "its fields are:");
                    for name in struct_.field_names() {
                        writeln!(s, "- `{}`", name);
                    }
                }
                s
            })),
        }
    }
}

impl Any {
    impl_track_caller_fn_any! {
        pub fn array_index (&self, i: Any)        -> Any => [*self, i] Expr::Decomposition(Decomposition::ArrayIndex);
        pub fn vector_index(&self, i: Any)        -> Any => [*self, i] Expr::Decomposition(Decomposition::VectorIndex);
        pub fn matrix_index(&self, i: Any)        -> Any => [*self, i] Expr::Decomposition(Decomposition::MatrixIndex);
        pub fn binding_array_index(&self, i: Any) -> Any => [*self, i] Expr::Decomposition(Decomposition::BindingArrayIndex);
        pub fn get_field(&self, name: CanonName)  -> Any => [*self]    Expr::Decomposition(Decomposition::StructureAccess(name));
        pub fn swizzle(&self, xyzw: VectorAccess) -> Any => [*self]    Expr::Decomposition(Decomposition::VectorAccess(xyzw));
    }

    #[track_caller]
    fn x(&self) -> Any { self.swizzle(VectorAccess::Swizzle1([Comp4::X])) }
    #[track_caller]
    fn y(&self) -> Any { self.swizzle(VectorAccess::Swizzle1([Comp4::Y])) }
    #[track_caller]
    fn z(&self) -> Any { self.swizzle(VectorAccess::Swizzle1([Comp4::Z])) }
    #[track_caller]
    fn w(&self) -> Any { self.swizzle(VectorAccess::Swizzle1([Comp4::W])) }

    /// applies swizzling to a 4 component vector to match the provided `len`.
    /// This applies either `.x` `.xy` `.xyz` or no operation at all
    #[track_caller]
    pub fn vec_x4_shrink(&self, len: ir::Len) -> Any {
        use Comp4::*;
        use VectorAccess::*;
        match len {
            Len::X1 => self.swizzle(Swizzle1([X])),
            Len::X2 => self.swizzle(Swizzle2([X, Y])),
            Len::X3 => self.swizzle(Swizzle3([X, Y, Z])),
            Len::X4 => *self,
        }
    }

    pub(crate) fn texture_sample_vec_shrink(self, sample_ty: TextureSampleUsageType) -> Any {
        match sample_ty.is_depth() {
            true => self,
            false => self.vec_x4_shrink(sample_ty.len()),
        }
    }
}

impl Any {
    /// get the x, y, z or w component of a vector
    #[track_caller]
    pub fn get_component(&self, comp: Comp4) -> Any { self.swizzle(VectorAccess::Swizzle1([comp])) }
}
