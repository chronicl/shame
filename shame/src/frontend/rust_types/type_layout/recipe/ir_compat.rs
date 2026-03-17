use crate::GpuLayout;

use super::*;

//     Conversions to ir types     //

/// Errors that can occur when converting IR types to recipe types.
#[derive(thiserror::Error, Debug, Clone)]
pub enum IRConversionError {
    /// Packed vectors do not exist in the shader type system.
    #[error("Type is or contains a packed vector, which does not exist in the shader type system.")]
    ContainsPackedVector,
    /// Struct field names must be unique in the shader type system.
    #[error("{0}")]
    DuplicateFieldName(#[from] DuplicateFieldNameError),
}

#[derive(Debug, Clone)]
pub struct DuplicateFieldNameError {
    pub struct_type: StructKind,
    pub first_occurence: usize,
    pub second_occurence: usize,
    pub use_color: bool,
}

impl std::error::Error for DuplicateFieldNameError {}

impl std::fmt::Display for DuplicateFieldNameError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (struct_name, sized_fields, last_unsized) = match &self.struct_type {
            StructKind::Sized(s) => (&s.name, s.fields(), None),
            StructKind::Unsized(s) => (&s.name, s.sized_fields.as_slice(), Some(&s.last_unsized)),
        };

        let indent = "  ";
        let is_duplicate = |i| self.first_occurence == i || self.second_occurence == i;
        let arrow = |i| match is_duplicate(i) {
            true => " <--",
            false => "",
        };
        let color = |f: &mut Formatter<'_>, i| {
            if (self.use_color && is_duplicate(i)) {
                set_color(f, Some("#508EE3"), false)
            } else {
                Ok(())
            }
        };
        let color_reset = |f: &mut Formatter<'_>, i| {
            if self.use_color && is_duplicate(i) {
                set_color(f, None, false)
            } else {
                Ok(())
            }
        };

        writeln!(
            f,
            "Type contains or is a struct with duplicate field names.\
            Field names must be unique in the shader type system.\n\
            The following struct contains duplicate field names:"
        )?;
        let header = writeln!(f, "struct {struct_name} {{");
        for (i, field) in sized_fields.iter().enumerate() {
            color(f, i)?;
            writeln!(f, "{indent}{}: {},{}", field.name, field.ty, arrow(i))?;
            color_reset(f, i)?;
        }
        if let Some(field) = last_unsized {
            let i = sized_fields.len();
            color(f, i)?;
            writeln!(f, "{indent}{}: {},{}", field.name, field.array, arrow(i))?;
            color_reset(f, i)?;
        }
        writeln!(f, "}}")?;

        Ok(())
    }
}

#[track_caller]
fn should_use_color() -> bool {
    Context::try_with(call_info!(), |ctx| ctx.settings().colored_error_messages).unwrap_or(false)
}

#[derive(Debug, Clone)]
pub enum StructKind {
    Sized(SizedStruct),
    Unsized(UnsizedStruct),
}

impl From<SizedStruct> for StructKind {
    fn from(value: SizedStruct) -> Self { StructKind::Sized(value) }
}
impl From<UnsizedStruct> for StructKind {
    fn from(value: UnsizedStruct) -> Self { StructKind::Unsized(value) }
}
