use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, FieldsNamed, Ident, Meta};

pub fn derive_to_glam_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let original_name = &input.ident;
    let cpu_name = Ident::new(&format!("{}Cpu", original_name), original_name.span());
    let vis = &input.vis;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Check for cpu_derive attribute
    let cpu_derive = input.attrs.iter().find_map(|attr| {
        if attr.path().is_ident("cpu_derive") {
            match &attr.meta {
                Meta::List(meta_list) => Some(meta_list.tokens.clone()),
                _ => None,
            }
        } else {
            None
        }
    });

    // Extract struct fields
    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(FieldsNamed { named, .. }) => named,
            _ => panic!("ToGlam can only be derived for structs with named fields"),
        },
        _ => panic!("ToGlam can only be derived for structs"),
    };

    // Generate CPU struct fields
    let cpu_fields: Vec<_> = fields
        .iter()
        .map(|field| {
            let field_name = &field.ident;
            let field_vis = &field.vis;
            let field_type = &field.ty;

            quote! {
                #field_vis #field_name: <#field_type as shame_utils::ToGlam>::GlamType
            }
        })
        .collect();

    // Generate the CPU struct
    let cpu_struct = if let Some(derive_meta) = cpu_derive {
        quote! {
            #[derive(#derive_meta)]
            #[repr(C)]
            #vis struct #cpu_name #impl_generics #where_clause {
                #(#cpu_fields,)*
            }
        }
    } else {
        quote! {
            #[repr(C)]
            #vis struct #cpu_name #impl_generics #where_clause {
                #(#cpu_fields,)*
            }
        }
    };

    // Generate ToGlam implementation for the original struct
    let to_glam_impl = quote! {
        impl #impl_generics shame_utils::ToGlam for #original_name #ty_generics #where_clause {
            type GlamType = #cpu_name #ty_generics;
        }
    };

    quote! {
        #cpu_struct
        #to_glam_impl
    }
    .into()
}
