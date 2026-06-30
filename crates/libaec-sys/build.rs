fn main() {
    if pkg_config::Config::new()
        .atleast_version("1.0")
        .probe("libaec")
        .is_ok()
    {
        return; // pkg-config found libaec and emitted the link directives
    }
    // libaec not found via pkg-config. Do not emit a link directive; the
    // szip feature in clawhdf5-format gates all actual FFI calls, so the
    // crate compiles and passes tests without libaec installed.
}
