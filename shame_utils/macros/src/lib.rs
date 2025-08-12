use proc_macro::TokenStream;

mod gpu_control_flow;
mod no_padding;

/// A procedural macro that transforms control flow constructs into shame control flow functions.
///
/// ## Modes:
///
/// - `#[gpu_control_flow(selective)]` - Only transforms control flow marked with `#[gpu]`
/// - `#[gpu_control_flow(all)]` - Transforms all control flow constructs
///
/// ## Supported constructs:
///
/// - `if condition { body }` → `sm::if_(condition, move || { body })`
/// - `if condition { body } else { else_body }` → `sm::if_else(condition, move || { body }, move || { else_body })`
/// - `for item in iter { body }` → `sm::for_(iter, move |item| { body })`
/// - `while condition { body }` → `sm::while_(condition, move || { body })`
///
/// ## Example:
/// ```
/// use shame as sm;
///
/// #[gpu_control_flow(selective)]
/// fn my_shader() {
///     #[gpu]
///     if some_condition {
///         // some code
///     }
///
///     #[gpu]
///     for item in some_iter {
///         // process item
///     }
/// }
///
/// #[gpu_control_flow(all)]
/// fn my_shader2() {
///     if some_condition {
///         // automatically transformed
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn gpu_control_flow(args: TokenStream, input: TokenStream) -> TokenStream {
    gpu_control_flow::gpu_control_flow_impl(args, input)
}

/// A derive macro that implements the `NoPadding` trait for structs.
///
/// This macro generates:
/// 1. A `NoPadding` implementation with a `LAYOUT` constant
/// 2. A compile-time padding check that panics with helpful error messages if padding is detected
///
/// ## Requirements:
/// - The struct must be `#[repr(C)]`
/// - All fields must implement `NoPadding`
///
/// ## Example:
/// ```
/// #[derive(NoPadding)]
/// #[repr(C)]
/// struct MyStruct {
///     a: F32x1,
///     b: F32x3,
/// }
/// ```
///
/// If padding is detected, the macro will generate a compile-time error with a suggestion
/// showing the struct definition with explicit padding fields.
#[proc_macro_derive(NoPadding)]
pub fn derive_no_padding(input: TokenStream) -> TokenStream { no_padding::derive_no_padding_impl(input) }
