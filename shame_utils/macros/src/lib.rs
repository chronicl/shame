use proc_macro::TokenStream;

mod gpu_control_flow;
mod no_padding;
mod to_glam;

/// A procedural attribute macro that transforms control flow constructs into shame control flow functions.
///
/// ## Modes:
///
/// - `#[gpu_control_flow]` or `#[gpu_control_flow(selective)]` – Only transforms control flow marked with `#[gpu]`
/// - `#[gpu_control_flow(all)]` – Transforms all control flow constructs
///
/// ## Supported constructs:
///
/// - `if condition { body }` → `shame::if_(condition, || { body })`
/// - `if condition { body } else { else_body }` → `shame::if_else(condition, || { body }, || { else_body })`
/// - `for item in iter { body }` → `shame::for_(iter, |item| { body })`
/// - `while condition { body }` → `shame::while_(condition, || { body })`
///
/// Additionally, any index expression `expr[i]` is rewritten to `expr.at(i)`.
///
/// ## Example (attribute form):
/// ```ignore
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

/// A function-like macro that transforms control flow constructs and index expressions
/// inside an inline block, without requiring a full `fn` item.
///
/// This only properly works with the `relaxed_control_flow` feature in shame enabled.
///
/// ## Supported constructs:
///
/// - `if condition { body }` → `shame::if_(condition, || { body })`
/// - `if condition { body } else { else_body }` → `shame::if_else(condition, || { body }, || { else_body })`
/// - `for item in range_iter { body }` → `shame::for_range(range_iter, |item| { body })`
/// - `while condition { body }` → `shame::while_(condition, || { body })`
/// - `expr[i] -> expr.at(i)`
/// - `expr = expr -> expr.set(expr)`
///
/// ## Example:
/// ```ignore
/// let result = gpu_code! {
///     if some_condition {
///         buf[idx] = 1.0.to_gpu();
///     } else {
///         buf[idx] = 0.0.to_gpu();
///     }
/// };
/// ```
#[proc_macro]
pub fn gpu_code(input: TokenStream) -> TokenStream {
    gpu_control_flow::gpu_control_flow_fn_impl(input, gpu_control_flow::TransformMode::All)
}

/// Same as `gpu_code`, but by marking control flow with `#[cpu]` it won't be transformed.
#[proc_macro]
pub fn gpu_code_selective_cpu(input: TokenStream) -> TokenStream {
    gpu_control_flow::gpu_control_flow_fn_impl(input, gpu_control_flow::TransformMode::SelectiveCpu)
}

/// Same as `gpu_code`, but control flow is only transformed if it's marked with `#[gpu]`.
#[proc_macro]
pub fn gpu_code_selective_gpu(input: TokenStream) -> TokenStream {
    gpu_control_flow::gpu_control_flow_fn_impl(input, gpu_control_flow::TransformMode::SelectiveGpu)
}

/// TODO(chronicl)
#[proc_macro_derive(NoPadding)]
pub fn derive_no_padding(input: TokenStream) -> TokenStream { no_padding::derive_no_padding_impl(input) }

/// TODO(chronicl)
#[proc_macro_derive(ToGlam, attributes(cpu_derive))]
pub fn derive_to_glam(input: TokenStream) -> TokenStream { to_glam::derive_to_glam_impl(input) }
