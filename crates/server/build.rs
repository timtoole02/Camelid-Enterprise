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
    println!("cargo:rustc-env=CAMELID_ENGINE_PIN={revision}");
}
