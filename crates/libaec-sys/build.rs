fn main() {
    if pkg_config::Config::new()
        .atleast_version("1.0")
        .probe("libaec")
        .is_ok()
    {
        return; // pkg-config found libaec and emitted the link directives
    }
    // Fallback: look for libaec.so / libaec.a in standard library paths.
    // libaec-dev on Debian/Ubuntu installs the library but omits the .pc file.
    let lib_dirs = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
        "/usr/local/lib",
        "/usr/local/lib/x86_64-linux-gnu",
    ];
    for dir in &lib_dirs {
        let so = std::path::Path::new(dir).join("libaec.so");
        let a = std::path::Path::new(dir).join("libaec.a");
        if so.exists() || a.exists() {
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-lib=aec");
            return;
        }
    }
    // libaec not found — szip feature will be unavailable but crate still compiles.
}
