use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Data, DeriveInput, Fields, FieldsNamed, Ident};

pub fn derive_no_padding_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Extract struct fields
    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(FieldsNamed { named, .. }) => named,
            _ => panic!("NoPadding can only be derived for structs with named fields"),
        },
        _ => panic!("NoPadding can only be derived for structs"),
    };

    if fields.is_empty() {
        panic!("NoPadding cannot be derived for empty structs");
    }

    // Generate field types and names
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();

    // Generate layout calculation
    let layout_calc = generate_layout_calculation(&field_types);

    // Generate padding check
    let padding_check = generate_padding_check2(name, &field_names, &field_types);

    let expanded = quote! {
        impl #impl_generics NoPadding for #name #ty_generics #where_clause {
            const LAYOUT: Layout = #layout_calc;
        }

        #padding_check
    };

    TokenStream::from(expanded)
}

fn generate_layout_calculation(field_types: &[&syn::Type]) -> proc_macro2::TokenStream {
    let mut layout_expr = quote! { Layout::from_align_size(1, Some(0)) };

    for field_type in field_types {
        layout_expr = quote! {
            #layout_expr.extend(<#field_type as NoPadding>::LAYOUT)
        };
    }

    layout_expr
}

#[derive(Clone, Copy)]
enum ConstIdent {
    Layout(usize),
    PaddingFieldCounter(usize),
    Padding(usize),
    FieldDecl(usize),
    PaddingDecls(usize),
}
use ConstIdent::*;

impl From<ConstIdent> for syn::Ident {
    fn from(ident: ConstIdent) -> syn::Ident {
        let (name, i) = match ident {
            Layout(i) => ("L", i),
            PaddingFieldCounter(i) => ("C", i),
            Padding(i) => ("P", i),
            FieldDecl(i) => ("FD", i),
            PaddingDecls(i) => ("PD", i),
        };
        syn::Ident::new(&format!("{name}{i}"), proc_macro2::Span::call_site())
    }
}
impl ToTokens for ConstIdent {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ident: syn::Ident = (*self).into();
        tokens.extend(ident.to_token_stream());
    }
}

fn generate_padding_check2(
    struct_name: &Ident,
    field_names: &[&Ident],
    field_types: &[&syn::Type],
) -> proc_macro2::TokenStream {
    let tab = "    ";
    let format_type = |ty: &syn::Type| ty.to_token_stream().to_string().replace(' ', "");

    #[cfg(feature = "pretty")]
    let format_field_decl = {
        let longest_field_name = field_names
            .iter()
            .map(|name| name.to_string().chars().count())
            .max()
            .unwrap_or(0);
        let longest_type_name = field_types
            .iter()
            .map(|ty| format_type(ty).chars().count())
            .max()
            .unwrap_or(0);

        let name_pad = longest_field_name + 1;
        let type_pad = longest_type_name + 1;

        move |name: &Ident, ty: &syn::Type| {
            let field_name = format!("{}:", name);
            let field_type = format!("{},", format_type(ty));
            format!("\n{tab}{:<name_pad$} {:<type_pad$}", field_name, field_type)
        }
    };
    #[cfg(not(feature = "pretty"))]
    let format_field_decl = |name: &Ident, ty: &syn::Type| format!("\n{tab}{}: {},", name, format_type(ty));

    let mut fields = Vec::new();
    for (i, (field_name, field_type)) in field_names.iter().zip(field_types.iter()).enumerate() {
        let layout = Layout(i); // layout of struct with all fields up to and including the ith field
        let field_decl = FieldDecl(i); // declaration of the ith field
        let padding_bytes = Padding(i); // padding bytes directly before the ith field
        let padding_field_counter = PaddingFieldCounter(i); // total number of padding fields before the ith field
        let padding_decls = PaddingDecls(i); // declarations of padding fields directly before the ith field
        let field_decl_str = format_field_decl(field_name, field_type);

        let tokens = if i == 0 {
            quote! {
                const #layout: ::shame_utils::Layout = <#field_type as NoPadding>::LAYOUT;
                const #field_decl: &str = #field_decl_str;
                const #padding_bytes: usize = 0;
                const #padding_field_counter: usize = 0;
                const #padding_decls: &str = "";
            }
        } else {
            let previous_layout = Layout(i - 1);
            let previous_padding_field_counter = PaddingFieldCounter(i - 1);

            quote! {
                const #layout: ::shame_utils::Layout = #previous_layout.extend(<#field_type as NoPadding>::LAYOUT);
                const #field_decl: &str = #field_decl_str;
                const #padding_bytes: usize = #layout.offset - #previous_layout.size.expect("non-last fields must be sized");
                const #padding_field_counter: usize = #previous_padding_field_counter + ::shame_utils::padding_to_padding_field_count::<#padding_bytes>();
                const #padding_decls: &str = ::shame_utils::padding_to_fields!(#padding_bytes, #previous_padding_field_counter);
            }
        };

        fields.push(tokens)
    }

    let last_field_index = fields
        .len()
        .checked_sub(1)
        .expect("shame structs must have at least one field");
    let last_layout = Layout(last_field_index);
    let last_field_counter = PaddingFieldCounter(last_field_index);
    let final_padding_bytes = Padding(fields.len());
    let final_padding_decls = PaddingDecls(fields.len());
    let final_padding_tokens = quote! {
        const #final_padding_bytes: usize = match (#last_layout.size_rounded_to_align(), #last_layout.size) {
            (Some(rounded_size), Some(size)) => rounded_size - size,
            _ => 0, // unsized structs have no padding at the end
        };
        const #final_padding_decls: &str = ::shame_utils::padding_to_fields!(#final_padding_bytes, #last_field_counter);
    };
    fields.push(final_padding_tokens);

    let padding_conditions: Vec<_> = (0..fields.len())
        .map(|i| {
            let padding_bytes = Padding(i);
            quote! { #padding_bytes != 0 }
        })
        .collect();

    let struct_start = format!(
        "\n\nImplicit padding detected. The struct definition with the implicit padding made explicit is:\n\n#[rustfmt::skip]\nstruct {} {{\n\n",
        struct_name
    );
    let struct_end = "}\n\n";

    let mut field_decls = Vec::<ConstIdent>::new();
    let original_fields_len = field_names.len();
    for i in 0..original_fields_len {
        field_decls.push(PaddingDecls(i));
        field_decls.push(FieldDecl(i));
    }
    field_decls.push(final_padding_decls);

    quote! {
        const _: () = {
            #(#fields)*

            if #(#padding_conditions)||* {
                const S: &str = ::shame_utils::const_concat_str![
                    #struct_start,
                    #(#field_decls),*,
                    #struct_end
                ];
                panic!("{}", S);
            }
        };
    }
}
