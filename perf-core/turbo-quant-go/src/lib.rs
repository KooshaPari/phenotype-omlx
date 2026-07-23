// turbo-quant-go — Rust workspace member for Go cgo binding.
// The actual Go source is in go-src/turboquant.go and links against turbo-quant-c.
pub fn placeholder() -> &'static str {
    "turbo-quant-go"
}
#[cfg(test)]
mod tests {
    #[test]
    fn go_placeholder() {
        assert_eq!(super::placeholder(), "turbo-quant-go");
    }
}
