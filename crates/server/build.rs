fn main() {
    let manifest_path =
        std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("Cargo.toml");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: toml::Value = toml::from_str(&manifest).unwrap();
    let revision = manifest["dependencies"]["camelid"]["rev"]
        .as_str()
        .expect("dependencies.camelid.rev must pin one engine revision");
    assert!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "dependencies.camelid.rev must be a full 40-character hexadecimal commit"
    );
    // Deliberately **not** `CAMELID_ENGINE_PIN`, and the prefix is the whole
    // reason. Cargo sets a build script's `rustc-env` variables in the
    // environment of the binaries it runs as well as in the compiler's, so a
    // `CAMELID_`-prefixed name here is a name present in the process environment
    // of every `cargo run` and `cargo test`. The lane's admission scan claims
    // that whole namespace deny-by-default, so it refused this one — a variable
    // this repository invented, that the pinned engine never reads, and that
    // nothing can unset because Cargo sets it after the shell is gone. The
    // effect was that `apply_deterministic` failed closed under Cargo:
    // `cargo run --bin camelid-enterprise -- serve` would not start, and the
    // model-backed conformance job could not run at all. Installed binaries were
    // unaffected, which is why it stayed invisible.
    //
    // Keeping it outside the namespace fixes that at the source rather than
    // teaching each test to scrub an environment it should not have to.
    println!("cargo:rustc-env=ENTERPRISE_ENGINE_PIN={revision}");
}
