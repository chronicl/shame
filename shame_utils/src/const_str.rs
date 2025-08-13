#[macro_export]
macro_rules! const_concat {
    ($($s:expr),*) => {{
        const STRINGS: &[&str] = &[$($crate::const_to_str!($s)),*];
        $crate::const_to_str!(STRINGS)
    }};
}

#[macro_export]
macro_rules! const_to_str {
    ($x:expr) => {{
        const LEN: usize = $crate::ToStr($x).bytes_len();
        const S: $crate::StrBuf<LEN> = $crate::ToStr($x).to_buf();
        S.as_str()
    }};
}

/// The length of the string produced by `const_concat`
#[macro_export]
macro_rules! const_len {
    ($($to_str:expr),*) => {
        $(
            $crate::ToStr($to_str).bytes_len() +
        )*
        0
    };
}

/// Writes the result of `const_concat` to a buffer at the location of pos.
/// This can be used in conjunction with `const_len` to implement ToStr(T)
/// for complex types `T`, see PaddingToFields as an example.
#[macro_export]
macro_rules! const_write {
    ($pos:ident, $buf:ident <- $($to_str:expr),*) => {{
        $(
            $pos += $crate::ToStr($to_str).write($pos, &mut $buf);
        )*
    }};
}

pub enum StrBuf<const N: usize> {
    Bytes([u8; N]),
    Str(&'static str),
}

impl<const N: usize> StrBuf<N> {
    pub const fn as_str(&self) -> &str {
        match self {
            StrBuf::Bytes(b) => match str::from_utf8(b) {
                Ok(s) => s,
                Err(_) => panic!("Invalid UTF-8 sequence in concatenated string"),
            },
            StrBuf::Str(s) => s,
        }
    }
}

pub struct ToStr<T>(pub T);

impl ToStr<&'static str> {
    pub const fn bytes_len(&self) -> usize { self.0.len() }

    pub const fn write(&self, mut pos: usize, buf: &mut [u8]) -> usize {
        let mut i = 0;
        let bytes = self.0.as_bytes();
        while i < bytes.len() {
            buf[pos] = bytes[i];
            pos += 1;
            i += 1;
        }
        bytes.len()
    }

    pub const fn to_buf<const N: usize>(&self) -> StrBuf<N> { StrBuf::Str(self.0) }
}

impl ToStr<usize> {
    pub const fn bytes_len(&self) -> usize {
        let mut n = self.0;

        let mut digits = 1;
        while n > 9 {
            n /= 10;
            digits += 1;
        }
        digits
    }

    pub const fn write(&self, pos: usize, buf: &mut [u8]) -> usize {
        let len = self.bytes_len();
        let mut i = pos + len - 1; // writing backwards
        let mut n = self.0;
        loop {
            buf[i] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
            i -= 1;
        }
        assert!(i == pos);
        len
    }

    /// Must be called with N = Self.output_len()
    pub const fn to_buf<const N: usize>(&self) -> StrBuf<N> {
        let mut buf = [0; N];
        self.write(0, &mut buf);
        StrBuf::Bytes(buf)
    }
}

impl ToStr<&[&'static str]> {
    pub const fn bytes_len(&self) -> usize {
        let mut len = 0;
        let mut i = 0;
        while i < self.0.len() {
            len += self.0[i].len();
            i += 1;
        }
        len
    }

    pub const fn write(&self, mut pos: usize, buf: &mut [u8]) -> usize {
        let start_pos = pos;
        let mut i = 0;
        while i < self.0.len() {
            pos += ToStr(self.0[i]).write(pos, buf);
            i += 1;
        }
        pos - start_pos
    }

    pub const fn to_buf<const N: usize>(&self) -> StrBuf<N> {
        let mut buf = [0u8; N];
        self.write(0, &mut buf);
        StrBuf::Bytes(buf)
    }
}
