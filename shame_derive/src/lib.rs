#![allow(non_snake_case, clippy::match_like_matches_macro)]

mod derive_layout;
mod util;

use derive_layout::WhichDerive;
use proc_macro::TokenStream;
use syn::parse_macro_input;

use syn::spanned::*;
use syn::token::Semi;
use syn::Data;
use syn::Fields;

/// implements [`GpuLayout`] and other traits for user defined structs
/// if all fields of the struct themselves implement [`GpuLayout`].
///
/// ## Example
/// ```
/// use shame as sm;
///
/// #[derive(sm::GpuLayout)]
/// struct PointLight {
///     position: sm::f32x4,
///     intensity: sm::f32x1,
/// }
/// ```
///
/// The derived memory layout follows to the WGSL struct member layout rules
/// described at
/// https://www.w3.org/TR/WGSL/#structure-member-layout
///
/// ## other traits
///
/// This macro conditionally implements
/// - [`ToGpuType`] where `Self::Gpu = Struct<Self>`
/// - [`GpuSized`]
///   if all fields are [`GpuSized`]
/// - [`GpuAligned`]
///   if all fields are [`GpuAligned`]
/// - [`NoBools`], [`NoHandles`], [`NoAtomics`]
///   if all fields implement those respective traits too
///
/// as well as some other traits that are used internally
///
/// ## custom alignment and size of fields
///
/// `align` and `size` attributes can be used in front of a struct field
/// to define a minimum alignment and byte-size requirement for that field.
/// ```
/// #[derive(sm::GpuLayout)]
/// struct PointLight {
///     #[align(16)] position: sm::f32x4,
///     #[size(16)] intensity: sm::f32x1,
/// }
///
/// #[derive(sm::GpuLayout)]
/// struct PointLight2 {
///     #[align(2)] // no effect, `position` already has a 16-byte alignment which makes it 2-byte aligned as well
///     position: sm::f32x4,
///     #[size(2)] // no effect, `intensity` is already larger, 4 bytes in size
///     intensity: sm::f32x1,
/// }
/// ```
///
/// ## Automatic Layout validation between Cpu and Gpu types
///
/// the `#[cpu(...)]` macro can be used to associate a Cpu type with a Gpu type
/// at the struct declaration level.
/// The equivalence of the two type's [`TypeLayout`]s is validated at pipeline
/// encoding time, as soon as the Gpu types is used in bindings, push-constants or
/// vertex buffers.
///
/// ```
/// #[derive(sm::CpuLayout)]
/// struct PointLightCpu {
///     angle: f32,
///     intensity: f32,
/// }
///
/// #[derive(sm::GpuLayout)]
/// #[cpu(PointLightCpu)] // associate PointLightGpu with PointLightCpu
/// struct PointLightGpu {
///     angle: sm::f32x1,
///     intensity: sm::f32x1,
/// }
/// ```
///
/// [`PackedVec`]: shame::packed::PackedVec
/// [`Ref`]: shame::Ref
/// [`Sampler`]: shame::Sampler
/// [`Texture`]: shame::Texture
/// [`StorageTexture`]: shame::StorageTexture
/// [`ToGpuType`]: shame::ToGpuType
/// [`GpuSized`]: shame::GpuSized
/// [`GpuAligned`]: shame::GpuAligned
/// [`NoBools`]: shame::NoBools
/// [`NoHandles`]: shame::NoHandles
/// [`NoAtomics`]: shame::NoAtomics
#[proc_macro_derive(GpuLayout, attributes(size, align, cpu, gpu_repr))]
pub fn derive_gpu_layout(input: TokenStream) -> TokenStream { derive_impl(WhichDerive::GpuLayout, input) }

#[proc_macro_derive(CpuLayout, attributes())]
pub fn derive_cpu_layout(input: TokenStream) -> TokenStream { derive_impl(WhichDerive::CpuLayout, input) }

fn derive_impl(which_derive: WhichDerive, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let span = input.span();

    match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(named_fields) => {
                derive_layout::impl_for_struct(which_derive, &input, &span, data_struct, named_fields)
            }
            Fields::Unnamed(_) | Fields::Unit => {
                Err(syn::Error::new(span, "Must be used on a struct with named fields"))
            }
        },
        Data::Union(_) | Data::Enum(_) => Err(syn::Error::new(span, "Must be used on a struct")),
    }
    .unwrap_or_else(|err| err.to_compile_error())
    .into()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TransformMode {
    Selective,
    All,
}

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
    let mut item_fn = parse_macro_input!(input as syn::ItemFn);

    // Parse the mode argument
    let mode = if args.is_empty() {
        TransformMode::Selective
    } else {
        let mode_input = parse_macro_input!(args as syn::Ident);
        match mode_input.to_string().as_str() {
            "selective" => TransformMode::Selective,
            "all" => TransformMode::All,
            _ => {
                return syn::Error::new(mode_input.span(), "Expected 'selective' or 'all'")
                    .to_compile_error()
                    .into();
            }
        }
    };

    // Transform the function body
    transform_block_stmts(&mut item_fn.block.as_mut().stmts, mode);

    quote::quote! { #item_fn }.into()
}

fn transform_block_stmts(stmts: &mut [syn::Stmt], mode: TransformMode) {
    for stmt in stmts.iter_mut() {
        transform_stmt(stmt, mode);
    }
}

fn transform_stmt(stmt: &mut syn::Stmt, mode: TransformMode) {
    match stmt {
        syn::Stmt::Expr(expr, semi) => transform_expr(expr, semi, mode),
        syn::Stmt::Local(local) => {
            if let Some(ref mut init) = local.init {
                transform_expr(&mut init.expr, &mut None, mode);
            }
        }
        syn::Stmt::Item(_) => {}
        syn::Stmt::Macro(_) => {}
    }
}

fn transform_expr(expr: &mut syn::Expr, semi: &mut Option<Semi>, mode: TransformMode) {
    // Check if this expression has a #[gpu] attribute
    let has_gpu_attr = match expr {
        syn::Expr::If(if_expr) => has_gpu_attribute(&if_expr.attrs),
        syn::Expr::ForLoop(for_expr) => has_gpu_attribute(&for_expr.attrs),
        syn::Expr::While(while_expr) => has_gpu_attribute(&while_expr.attrs),
        _ => false,
    };

    // Determine if we should transform based on mode and gpu attribute
    let should_transform = match mode {
        TransformMode::Selective => has_gpu_attr,
        TransformMode::All => match expr {
            syn::Expr::If(_) | syn::Expr::ForLoop(_) | syn::Expr::While(_) => true,
            _ => false,
        },
    };

    if should_transform {
        match expr {
            syn::Expr::If(if_expr) => {
                let transformed = transform_if_expr(if_expr.clone(), semi);
                *expr = syn::parse2(transformed).expect("failed to parse");
            }
            syn::Expr::ForLoop(for_expr) => {
                let transformed = transform_for_expr(for_expr.clone(), semi);
                *expr = syn::parse2(transformed).expect("failed to parse");
            }
            syn::Expr::While(while_expr) => {
                let transformed = transform_while_expr(while_expr.clone(), semi);
                *expr = syn::parse2(transformed).expect("failed to parse");
            }
            _ => {}
        }
    } else {
        // Recursively transform nested expressions
        match expr {
            syn::Expr::Block(block_expr) => {
                transform_block_stmts(&mut block_expr.block.stmts, mode);
            }
            syn::Expr::If(if_expr) => {
                transform_expr(&mut if_expr.cond, &mut None, mode);
                transform_block_stmts(&mut if_expr.then_branch.stmts, mode);
                if let Some((_, else_expr)) = &mut if_expr.else_branch {
                    transform_expr(else_expr, &mut None, mode);
                }
            }
            syn::Expr::ForLoop(for_expr) => {
                transform_expr(&mut for_expr.expr, &mut None, mode);
                transform_block_stmts(&mut for_expr.body.stmts, mode);
            }
            syn::Expr::While(while_expr) => {
                transform_expr(&mut while_expr.cond, &mut None, mode);
                transform_block_stmts(&mut while_expr.body.stmts, mode);
            }
            _ => {}
        }
    }
}

fn has_gpu_attribute(attrs: &[syn::Attribute]) -> bool { attrs.iter().any(|attr| attr.path().is_ident("gpu")) }

fn transform_if_expr(mut if_expr: syn::ExprIf, semi: &mut Option<Semi>) -> proc_macro2::TokenStream {
    // Remove the #[gpu] attribute
    if_expr.attrs.retain(|attr| !attr.path().is_ident("gpu"));

    let cond = &if_expr.cond;
    let then_branch = &if_expr.then_branch;

    ensure_semicolon(semi);
    if let Some((_, else_branch)) = &if_expr.else_branch {
        quote::quote! {
            shame::if_else(#cond, move || #then_branch, move || #else_branch)
        }
    } else {
        quote::quote! {
            shame::if_(#cond, move || #then_branch)
        }
    }
}

fn transform_for_expr(mut for_expr: syn::ExprForLoop, semi: &mut Option<Semi>) -> proc_macro2::TokenStream {
    // Remove the #[gpu] attribute
    for_expr.attrs.retain(|attr| !attr.path().is_ident("gpu"));

    let pat = &for_expr.pat;
    let expr = &for_expr.expr;
    let body = &for_expr.body;

    ensure_semicolon(semi);
    quote::quote! {
        shame::for_(#expr, move |#pat| #body)
    }
}

fn transform_while_expr(mut while_expr: syn::ExprWhile, semi: &mut Option<Semi>) -> proc_macro2::TokenStream {
    // Remove the #[gpu] attribute
    while_expr.attrs.retain(|attr| !attr.path().is_ident("gpu"));

    let cond = &while_expr.cond;
    let body = &while_expr.body;

    ensure_semicolon(semi);
    quote::quote! {
        shame::while_(#cond, move || #body)
    }
}

fn ensure_semicolon(semi: &mut Option<Semi>) {
    if semi.is_none() {
        *semi = Some(Semi::default());
    }
}
