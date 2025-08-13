use proc_macro::TokenStream;
use syn::{parse_macro_input, token::Semi};

#[derive(Debug, Clone, Copy, PartialEq)]
enum TransformMode {
    Selective,
    All,
}

pub(crate) fn gpu_control_flow_impl(args: TokenStream, input: TokenStream) -> TokenStream {
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
        TransformMode::All => matches!(expr, syn::Expr::If(_) | syn::Expr::ForLoop(_) | syn::Expr::While(_)),
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
