use super::*;

// TODO(chronicl) This impl is marker::MaybeInvalid, because it's mostly used for
// CpuTypeLayout = TypeLayout<marker::MaybeInvalid> construction. Consider making some of
// these methods available to TypeLayout<marker::Valid>.
impl TypeLayout<marker::MaybeInvalid> {
    pub(crate) fn new(byte_size: Option<u64>, byte_align: u64, kind: TypeLayoutSemantics) -> Self {
        Self {
            byte_size,
            byte_align: byte_align.into(),
            kind,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn from_rust_sized<T: Sized>(kind: TypeLayoutSemantics) -> Self {
        Self::new(Some(size_of::<T>() as u64), align_of::<T>() as u64, kind)
    }

    pub(crate) fn first_line_of_display_with_ellipsis(&self) -> String {
        let string = format!("{}", self);
        string.split_once('\n').map(|(s, _)| format!("{s}…")).unwrap_or(string)
    }
}

impl StructLayoutBuilder<marker::MaybeInvalid> {}
