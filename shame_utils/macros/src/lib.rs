use proc_macro::TokenStream;

mod gpu_control_flow;
mod no_padding;
mod to_glam;

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

/// TODO(chronicl)
#[proc_macro_derive(NoPadding)]
pub fn derive_no_padding(input: TokenStream) -> TokenStream { no_padding::derive_no_padding_impl(input) }

/// TODO(chronicl)
#[proc_macro_derive(ToGlam, attributes(cpu_derive))]
pub fn derive_to_glam(input: TokenStream) -> TokenStream { to_glam::derive_to_glam_impl(input) }
