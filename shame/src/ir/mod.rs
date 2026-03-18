pub(crate) mod expr;
pub(crate) mod ir_type;
pub(crate) mod pipeline;
pub(crate) mod recording;

pub use recording::CallInfo;
pub(crate) use recording::Node;

pub use ir_type::AccessMode;
pub use ir_type::AccessModeReadable;
pub use ir_type::AddressSpace;
pub use ir_type::AlignedType;
pub use ir_type::HandleType;

pub use ir_type::StoreType;
pub use ir_type::Type;

pub use ir_type::PackedBitsPerComponent;
pub use ir_type::PackedFloat;
pub use ir_type::PackedScalarType;
pub use ir_type::PackedVector;

pub use ir_type::StructRegistry;
pub use ir_type::StructDef;
pub use ir_type::StructDefinitionError;
pub use ir_type::StructureFieldNamesMustBeUnique;

pub use expr::AtomicModify;
pub use expr::Comp4;
pub use expr::CompoundOp;
pub use expr::GradPrecision;
pub use expr::VectorAccess;

pub use ir_type::ChannelFormatShaderType;
pub use ir_type::FragmentShadingRate;
pub use ir_type::SamplesPerPixel;
pub use ir_type::TextureAspect;
pub use ir_type::TextureFormatId;
pub use ir_type::TextureFormatWrapper;
pub use ir_type::TextureSampleUsageType;

pub use ir_type::layout_type::Vector;
pub use ir_type::layout_type::Matrix;
pub use ir_type::layout_type::Atomic;
pub use ir_type::layout_type::SizedArray;
pub use ir_type::layout_type::SizedStruct;
pub use ir_type::layout_type::SizedType;
pub use ir_type::layout_type::RuntimeSizedArray;
pub use ir_type::layout_type::UnsizedStruct;
pub use ir_type::layout_type::LayoutType;
pub use ir_type::layout_type::RuntimeSizedArrayField;
pub use ir_type::layout_type::SizedField;
pub use ir_type::layout_type::StructKind;
pub use ir_type::layout_type::StructKindRef;
pub use ir_type::layout_type::FieldOffsets;
pub use ir_type::layout_type::Repr;

pub use ir_type::layout_type::Len;
pub use ir_type::layout_type::Len2;
pub use ir_type::layout_type::LenEven;
pub use ir_type::layout_type::ScalarConstant;
pub use ir_type::layout_type::ScalarType;
pub use ir_type::layout_type::ScalarTypeFp;
pub use ir_type::layout_type::ScalarTypeInteger;
pub use ir_type::layout_type::CanonName;
pub use ir_type::layout_type::FieldOptions;

pub(crate) use ir_type::layout_type::type_layout;
