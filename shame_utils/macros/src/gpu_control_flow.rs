use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{
    Attribute,
    parse::{Parse, ParseStream},
    parse_macro_input,
    spanned::Spanned as _,
    token::Semi,
};


#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TransformMode {
    SelectiveGpu,
    SelectiveCpu { transform_let_mut: bool },
    All { transform_let_mut: bool },
}

impl TransformMode {
    pub fn should_transform_let_mut(&self) -> bool {
        match self {
            TransformMode::SelectiveGpu => false,
            TransformMode::SelectiveCpu { transform_let_mut } => *transform_let_mut,
            TransformMode::All { transform_let_mut } => *transform_let_mut,
        }
    }
}

pub(crate) fn gpu_control_flow_impl(args: TokenStream, input: TokenStream) -> TokenStream {
    fn parse_mode(args: TokenStream) -> Result<TransformMode, syn::Error> {
        if args.is_empty() {
            return Ok(TransformMode::SelectiveGpu);
        }

        struct ModeArgs {
            idents: syn::punctuated::Punctuated<syn::Ident, syn::Token![,]>,
        }
        impl Parse for ModeArgs {
            fn parse(input: ParseStream) -> syn::Result<Self> {
                Ok(ModeArgs {
                    idents: syn::punctuated::Punctuated::parse_terminated(input)?,
                })
            }
        }

        let mode_args: ModeArgs = syn::parse(args)?;
        let mut idents = mode_args.idents.iter();

        let first = idents
            .next()
            .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "expected `selective` or `all`"))?;

        let skip_let_mut = idents.any(|i| i == "skip_let_mut");

        match first.to_string().as_str() {
            "selective" => Ok(TransformMode::SelectiveGpu),
            // for attribute macros we always don't transform let mut
            "all" => Ok(TransformMode::All {
                transform_let_mut: !skip_let_mut,
            }),
            _ => Err(syn::Error::new(first.span(), "expected `selective` or `all`")),
        }
    }


    let mut item_fn = parse_macro_input!(input as syn::ItemFn);

    let mode = match parse_mode(args) {
        Ok(m) => m,
        Err(e) => return e.to_compile_error().into(),
    };

    transform_block_stmts(&mut item_fn.block.stmts, mode);

    quote! { #item_fn }.into()
}


pub(crate) fn gpu_control_flow_fn_impl(input: TokenStream, mode: TransformMode) -> TokenStream {
    struct StmtList(Vec<syn::Stmt>);
    impl Parse for StmtList {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let mut stmts = Vec::new();
            while !input.is_empty() {
                stmts.push(input.parse()?);
            }
            Ok(StmtList(stmts))
        }
    }

    let mut stmts = parse_macro_input!(input as StmtList).0;
    transform_block_stmts(&mut stmts, mode);
    quote! { #(#stmts)* }.into()
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
            if mode.should_transform_let_mut() && should_transform(&local.attrs, mode) {
                // transform_let_mut `let mut pat = expr` -> `let pat = ::shame::Cell::new(expr)`
                if let syn::Pat::Ident(ref mut pat_ident) = local.pat {
                    if let Some(mut_token) = pat_ident.mutability.take() {
                        let span = mut_token.span;
                        if let Some(ref mut init) = local.init {
                            transform_expr(&mut init.expr, &mut None, mode);
                            if let Some((_, ref mut diverge)) = init.diverge {
                                transform_expr(diverge, &mut None, mode);
                            }
                            let inner = init.expr.clone();
                            init.expr = Box::new(
                                syn::parse2(quote_spanned! { span => ::shame::Cell::new(#inner) })
                                    .expect("failed to parse Cell::new wrapping"),
                            );
                            return;
                        }
                    }
                }
            }
            if let Some(ref mut init) = local.init {
                transform_expr(&mut init.expr, &mut None, mode);
                if let Some((_, ref mut diverge)) = init.diverge {
                    transform_expr(diverge, &mut None, mode);
                }
            }
        }
        syn::Stmt::Item(_) | syn::Stmt::Macro(_) => {}
    }
}

/// - Control-flow rewriting  (if / for / while -> shame::*)
/// - Index rewriting         (expr[i] -> expr.at(i))
/// - Assignment rewriting    (expr = expr -> expr.set(expr))
/// - Comparison rewriting    (a < b -> a.less_than(b), a <= b -> a.less_eq(b), ...)
/// - Assign-op rewriting     (a += b -> a.set_add(b), a -= b -> a.set_sub(b), ...)
/// - Generic recursive descent into everything else
fn transform_expr(expr: &mut syn::Expr, semi: &mut Option<Semi>, mode: TransformMode) {
    match expr {
        syn::Expr::If(e) if should_transform(&e.attrs, mode) => {
            transform_expr(&mut e.cond, &mut None, mode);
            transform_block_stmts(&mut e.then_branch.stmts, mode);

            let span = e.span();
            let cond = &e.cond;
            let then_branch = &e.then_branch;
            let ts = match &mut e.else_branch {
                Some((_, else_expr)) => {
                    transform_expr(else_expr, &mut None, mode);
                    quote_spanned! { span => ::shame::if_else(#cond, || #then_branch, || #else_expr) }
                }
                None => quote_spanned! { span => ::shame::if_(#cond, || #then_branch) },
            };

            ensure_semicolon(semi);
            *expr = syn::parse2(ts).expect("failed to parse transformed if");
        }
        syn::Expr::ForLoop(e) if should_transform(&e.attrs, mode) => {
            transform_expr(&mut e.expr, &mut None, mode);
            transform_block_stmts(&mut e.body.stmts, mode);

            let span = e.span();
            let pat = &e.pat;
            let iter = &e.expr;
            let body = &e.body;

            ensure_semicolon(semi);
            let ts = quote_spanned! { span => ::shame::for_range(#iter, |#pat| #body) };
            *expr = syn::parse2(ts).expect("failed to parse transformed for");
        }
        syn::Expr::While(e) if should_transform(&e.attrs, mode) => {
            transform_expr(&mut e.cond, &mut None, mode);
            transform_block_stmts(&mut e.body.stmts, mode);

            let span = e.span();
            let cond = &e.cond;
            let body = &e.body;

            ensure_semicolon(semi);
            let ts = quote_spanned! { span => ::shame::while_(|| #cond, || #body) };
            *expr = syn::parse2(ts).expect("failed to parse transformed while");
        }
        syn::Expr::Index(idx) if should_transform(&idx.attrs, mode) => {
            // expr[i] -> expr.at(i)
            let span = idx.bracket_token.span.join();
            transform_expr(&mut idx.expr, &mut None, mode);
            transform_expr(&mut idx.index, &mut None, mode);

            let base = &*idx.expr;
            let index = &*idx.index;
            let ts = quote_spanned! { span => (#base).at((#index)) };
            *expr = syn::parse2(ts).expect("failed to parse .at() call");
        }
        syn::Expr::Assign(assign) if should_transform(&assign.attrs, mode) => {
            // expr = expr -> expr.set(expr)
            let span = assign.eq_token.span;
            transform_expr(&mut assign.left, &mut None, mode);
            transform_expr(&mut assign.right, &mut None, mode);

            let left = &*assign.left;
            let right = &*assign.right;
            let ts = quote_spanned! { span => (#left).set((#right)) };
            *expr = syn::parse2(ts).expect("failed to parse .set() call");
        }
        syn::Expr::Binary(bin) if should_transform(&bin.attrs, mode) => {
            transform_expr(&mut bin.left, &mut None, mode);
            transform_expr(&mut bin.right, &mut None, mode);

            // comparison / equality operators -> method calls
            // assign-op operators -> method calls
            let method = match bin.op {
                syn::BinOp::Lt(_) => Some("less_than"),
                syn::BinOp::Le(_) => Some("less_eq"),
                syn::BinOp::Gt(_) => Some("greater_than"),
                syn::BinOp::Ge(_) => Some("greater_eq"),
                syn::BinOp::Eq(_) => Some("equals"),
                syn::BinOp::Ne(_) => Some("not_equals"),
                syn::BinOp::AddAssign(_) => Some("set_add"),
                syn::BinOp::SubAssign(_) => Some("set_sub"),
                syn::BinOp::MulAssign(_) => Some("set_mul"),
                syn::BinOp::DivAssign(_) => Some("set_div"),
                syn::BinOp::RemAssign(_) => Some("set_rem"),
                syn::BinOp::BitAndAssign(_) => Some("set_bitand"),
                syn::BinOp::BitOrAssign(_) => Some("set_bitor"),
                syn::BinOp::BitXorAssign(_) => Some("set_bitxor"),
                syn::BinOp::ShlAssign(_) => Some("set_shl"),
                syn::BinOp::ShrAssign(_) => Some("set_shr"),
                _ => None,
            };
            if let Some(method_name) = method {
                let span = bin.op.span();
                let left = &*bin.left;
                let right = &*bin.right;
                let method_ident = syn::Ident::new(method_name, span);

                let ts = quote_spanned! { span => (#left).#method_ident((#right)) };
                *expr = syn::parse2(ts).expect("failed to parse binary method call");
            }
        }
        syn::Expr::Return(return_expr) if should_transform(&return_expr.attrs, mode) => {
            let span = return_expr.return_token.span;

            if let Some(ref mut e) = return_expr.expr {
                transform_expr(e, &mut None, mode);
                let ts = quote_spanned! { span => ::shame::return_value(#e) };
                *expr = syn::parse2(ts).expect("failed to parse transformed return");
            } else {
                let ts = quote_spanned! { span => ::shame::return_() };
                *expr = syn::parse2(ts).expect("failed to parse transformed return");
            }
        }
        syn::Expr::Break(break_expr) if should_transform(&break_expr.attrs, mode) => {
            let span = break_expr.break_token.span;

            if break_expr.expr.is_some() {
                panic!("breaking with a value is not supported in GPU control flow transformations.");
            } else {
                let ts = quote_spanned! { span => ::shame::break_() };
                *expr = syn::parse2(ts).expect("failed to parse transformed break");
            }
        }
        // Non-transforms
        syn::Expr::Block(e) => {
            transform_block_stmts(&mut e.block.stmts, mode);
        }
        syn::Expr::If(e) => {
            transform_expr(&mut e.cond, &mut None, mode);
            transform_block_stmts(&mut e.then_branch.stmts, mode);
            if let Some((_, else_expr)) = &mut e.else_branch {
                transform_expr(else_expr, &mut None, mode);
            }
        }
        syn::Expr::ForLoop(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
            transform_block_stmts(&mut e.body.stmts, mode);
        }
        syn::Expr::While(e) => {
            transform_expr(&mut e.cond, &mut None, mode);
            transform_block_stmts(&mut e.body.stmts, mode);
        }
        syn::Expr::Loop(e) => {
            transform_block_stmts(&mut e.body.stmts, mode);
        }
        syn::Expr::Index(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
            transform_expr(&mut e.index, &mut None, mode);
        }
        syn::Expr::Closure(e) => {
            transform_expr(&mut e.body, &mut None, mode);
        }
        syn::Expr::Call(e) => {
            transform_expr(&mut e.func, &mut None, mode);
            for arg in &mut e.args {
                transform_expr(arg, &mut None, mode);
            }
        }
        syn::Expr::MethodCall(e) => {
            transform_expr(&mut e.receiver, &mut None, mode);
            for arg in &mut e.args {
                transform_expr(arg, &mut None, mode);
            }
        }
        syn::Expr::Binary(e) => {
            transform_expr(&mut e.left, &mut None, mode);
            transform_expr(&mut e.right, &mut None, mode);
        }
        syn::Expr::Unary(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
        }
        syn::Expr::Assign(e) => {
            transform_expr(&mut e.left, &mut None, mode);
            transform_expr(&mut e.right, &mut None, mode);
        }
        syn::Expr::Field(e) => {
            transform_expr(&mut e.base, &mut None, mode);
        }
        syn::Expr::Reference(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
        }
        syn::Expr::Paren(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
        }
        syn::Expr::Cast(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
        }
        syn::Expr::Try(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
        }
        syn::Expr::Await(e) => {
            transform_expr(&mut e.base, &mut None, mode);
        }
        syn::Expr::Return(e) => {
            if let Some(ref mut val) = e.expr {
                transform_expr(val, &mut None, mode);
            }
        }
        syn::Expr::Break(e) => {
            if let Some(ref mut val) = e.expr {
                transform_expr(val, &mut None, mode);
            }
        }
        syn::Expr::Yield(e) => {
            if let Some(ref mut val) = e.expr {
                transform_expr(val, &mut None, mode);
            }
        }
        syn::Expr::Tuple(e) => {
            for elem in &mut e.elems {
                transform_expr(elem, &mut None, mode);
            }
        }
        syn::Expr::Array(e) => {
            for elem in &mut e.elems {
                transform_expr(elem, &mut None, mode);
            }
        }
        syn::Expr::Repeat(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
            transform_expr(&mut e.len, &mut None, mode);
        }
        syn::Expr::Struct(e) => {
            for field in &mut e.fields {
                transform_expr(&mut field.expr, &mut None, mode);
            }
            if let Some(ref mut rest) = e.rest {
                transform_expr(rest, &mut None, mode);
            }
        }
        syn::Expr::Match(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
            for arm in &mut e.arms {
                if let Some((_, ref mut guard)) = arm.guard {
                    transform_expr(guard, &mut None, mode);
                }
                transform_expr(&mut arm.body, &mut None, mode);
            }
        }
        syn::Expr::Range(e) => {
            if let Some(ref mut start) = e.start {
                transform_expr(start, &mut None, mode);
            }
            if let Some(ref mut end) = e.end {
                transform_expr(end, &mut None, mode);
            }
        }
        syn::Expr::Let(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
        }
        syn::Expr::Async(e) => {
            transform_block_stmts(&mut e.block.stmts, mode);
        }
        syn::Expr::Const(e) => {
            transform_block_stmts(&mut e.block.stmts, mode);
        }
        syn::Expr::Unsafe(e) => {
            transform_block_stmts(&mut e.block.stmts, mode);
        }
        syn::Expr::TryBlock(e) => {
            transform_block_stmts(&mut e.block.stmts, mode);
        }
        syn::Expr::Group(e) => {
            transform_expr(&mut e.expr, &mut None, mode);
        }
        syn::Expr::Continue(_) |
        syn::Expr::Infer(_) |
        syn::Expr::Lit(_) |
        syn::Expr::Macro(_) |
        syn::Expr::Path(_) |
        syn::Expr::RawAddr(_) |
        syn::Expr::Verbatim(_) => {}
        // syn::Expr is non-exhaustive
        _ => {}
    }
}

fn should_transform(attrs: &[syn::Attribute], mode: TransformMode) -> bool {
    match mode {
        TransformMode::SelectiveGpu => has_gpu_attribute(attrs),
        TransformMode::SelectiveCpu { .. } => has_cpu_attribute(attrs),
        TransformMode::All { .. } => true,
    }
}

fn has_gpu_attribute(attrs: &[syn::Attribute]) -> bool { attrs.iter().any(|attr| attr.path().is_ident("gpu")) }
fn has_cpu_attribute(attrs: &[syn::Attribute]) -> bool { attrs.iter().any(|attr| attr.path().is_ident("cpu")) }

fn ensure_semicolon(semi: &mut Option<Semi>) {
    if semi.is_none() {
        *semi = Some(Semi::default());
    }
}
