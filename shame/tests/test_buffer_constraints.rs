#[test]
fn buffer() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/buffer_constraints_fail/*.rs");
}

#[test]
fn atomic_in_storage() { let _: shame::Buffer<shame::Atomic<u32>, shame::mem::Storage> = todo!(); }

#[test]
fn readwrite_storage() {
    let _: shame::Buffer<shame::Ref<shame::f32x1>, shame::mem::Storage, shame::ReadWrite> = todo!();
}
