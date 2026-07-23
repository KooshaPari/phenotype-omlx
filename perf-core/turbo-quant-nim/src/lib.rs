// turbo-quant-nim — Rust workspace member for Nim dynlib binding.
pub fn placeholder() -> &'static str {
    "turbo-quant-nim"
}
#[cfg(test)]
mod tests {
    #[test]
    fn nim_placeholder() {
        assert_eq!(super::placeholder(), "turbo-quant-nim");
    }
}
