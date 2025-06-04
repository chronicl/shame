use shame as sm;
use sm::{mem, f32x1, Ref, Array, Size};

#[test]
fn buffer() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/buffer_constraints_fail/*.rs");
}

#[test]
fn atomic_in_storage() { let _: sm::Buffer<sm::Atomic<u32>, mem::Storage> = todo!(); }

#[test]
fn readwrite_storage() { let _: sm::Buffer<Ref<f32x1>, mem::Storage, sm::ReadWrite> = todo!(); }

#[test]
fn sized_array() { let _: sm::Buffer<Array<f32x1, Size<1>>, mem::Storage> = todo!(); }

#[test]
fn runtime_sized_array() { let _: sm::Buffer<Ref<Array<f32x1>>, mem::Storage> = todo!(); }
