use proc_macro::TokenStream;
use quote::quote;
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
    let padding_check = generate_padding_check(name, &field_names, &field_types);

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

/// The first field of the struct has index 1, index 0 is used for initial values
enum Const {
    /// Layout of the struct up to the ith field
    Layout,
    /// Padding field counter for the padding fields after the ith field
    PaddingFieldCounter,
    /// Padding byte size after the ith field
    Padding,
    /// Field declaration of the ith field
    FieldDecl,
    /// Padding declarations after the ith field
    PaddingDecls,
}

impl Const {
    pub fn get_ident(&self, i: usize) -> syn::Ident {
        let letter = match self {
            Const::Layout => "L",
            Const::PaddingFieldCounter => "C",
            Const::Padding => "P",
            Const::FieldDecl => "FD",
            Const::PaddingDecls => "PD",
        };
        syn::Ident::new(&format!("{letter}{i}"), proc_macro2::Span::call_site())
    }
}

fn generate_padding_check(
    struct_name: &Ident,
    field_names: &[&Ident],
    field_types: &[&syn::Type],
) -> proc_macro2::TokenStream {
    let struct_name_str = struct_name.to_string();

    // Generate layout calculations for each step
    let mut layouts = Vec::new();
    let mut padding_calculations = Vec::new();
    let mut field_counters = Vec::new();
    let mut struct_string_parts = Vec::new();

    // Initial values
    let initial_layout_ident = Const::Layout.get_ident(0);
    layouts.push(quote! {
        const #initial_layout_ident: Layout = Layout::from_align_size(1, Some(0));
    });
    let initial_field_counter_ident = Const::PaddingFieldCounter.get_ident(0);
    field_counters.push(quote! {
        const #initial_field_counter_ident: usize = 0;
    });

    // Generate calculations for each field
    for (i, (field_name, field_type)) in field_names.iter().zip(field_types.iter()).enumerate() {
        let layout_prev = Const::Layout.get_ident(i);
        let layout_curr = Const::Layout.get_ident(i + 1);
        let padding_curr = Const::Padding.get_ident(i);
        let counter_curr = Const::PaddingFieldCounter.get_ident(i);
        let field_curr = Const::FieldDecl.get_ident(i + 1);
        let padding_decl_curr = Const::PaddingDecls.get_ident(i + 1);

        layouts.push(quote! {
            const #layout_curr: Layout = #layout_prev.extend(<#field_type as NoPadding>::LAYOUT);
        });

        padding_calculations.push(quote! {
            const #padding_curr: usize = #layout_curr.offset - #layout_prev.size.unwrap();
        });

        if i > 0 {
            let counter_prev = Const::PaddingFieldCounter.get_ident(i - 1);
            let padding_prev = Const::Padding.get_ident(i - 1);
            field_counters.push(quote! {
                const #counter_curr: usize = #counter_prev + shame_utils::padding_to_padding_field_count::<#padding_prev>();
            });
        }

        let field_name_str = field_name.to_string();
        let field_line = format!("    {}: {},", field_name_str, quote!(#field_type));
        struct_string_parts.push(quote! {
            const #field_curr: &str = #field_line;
        });

        struct_string_parts.push(quote! {
            const #padding_decl_curr: &str = shame_utils::padding_to_fields!(#padding_curr, #counter_curr);
        });
    }

    // Final padding calculation
    let last_layout = Const::Layout.get_ident(field_names.len());
    let final_padding = Const::Padding.get_ident(field_names.len());
    let final_counter = Const::PaddingFieldCounter.get_ident(field_names.len());

    padding_calculations.push(quote! {
        const #final_padding: usize = #last_layout.size_rounded_to_align().unwrap() - #last_layout.size.unwrap();
    });

    let prev_counter = Const::PaddingFieldCounter.get_ident(field_names.len() - 1);
    let prev_padding = Const::Padding.get_ident(field_names.len() - 1);
    field_counters.push(quote! {
        const #final_counter: usize = #prev_counter + shame_utils::padding_to_padding_field_count::<#prev_padding>();
    });

    let final_sp = Const::PaddingDecls.get_ident(field_names.len() + 1);
    struct_string_parts.push(quote! {
        const #final_sp: &str = shame_utils::padding_to_fields!(#final_padding, #final_counter);
    });

    // Generate padding condition
    let padding_conditions: Vec<_> = (0..=field_names.len())
        .map(|i| {
            let p = Const::Padding.get_ident(i);
            quote! { #p != 0 }
        })
        .collect();

    // Generate struct string concatenation
    let struct_header = format!("\n#[rustfmt::skip]\nstruct {} {{", struct_name_str);
    let mut concat_parts = vec![
        quote! { "\n\nImplicit padding detected. The struct definition with the implicit padding made explicit is:\n" },
        quote! { #struct_header },
    ];

    for i in 0..field_names.len() {
        let s = Const::PaddingDecls.get_ident(i + 1);
        let s_field = Const::FieldDecl.get_ident(i + 1);
        concat_parts.push(quote! { #s });
        concat_parts.push(quote! { #s_field });
    }

    let final_padding_decl = Const::PaddingDecls.get_ident(field_names.len() + 1);
    concat_parts.push(quote! { #final_padding_decl });
    concat_parts.push(quote! { "}\n" });

    quote! {
        const _: () = {
            #(#layouts)*
            #(#padding_calculations)*
            #(#field_counters)*
            #(#struct_string_parts)*

            if #(#padding_conditions)||* {
                const S: &str = shame_utils::const_str::concat![
                    #(#concat_parts),*
                ];
                panic!("{}", S);
            }
        };
    }
}
