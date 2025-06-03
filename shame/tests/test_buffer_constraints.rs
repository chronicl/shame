use shame::{self as sm, ReadWrite, mem, f32x1, Array, Size};

#[test]
fn buffer() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/buffer_constraints_fail/*.rs");
}

#[test]
fn atomic_in_storage() { let _: sm::Buffer<sm::Atomic<u32>, mem::Storage> = todo!(); }

#[test]
fn readwrite_storage() { let _: sm::Buffer<f32x1, mem::Storage, sm::ReadWrite> = todo!(); }

#[test]
fn sized_array() { let _: sm::Buffer<Array<f32x1, Size<1>>, mem::Storage> = todo!(); }

#[test]
fn runtime_sized_array() { let _: sm::Buffer<Array<f32x1>, mem::Storage, ReadWrite> = todo!(); }
