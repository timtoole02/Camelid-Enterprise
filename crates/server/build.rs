fn main() {
    let manifest_path =
        std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("Cargo.toml");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: toml::Value = toml::from_str(&manifest).unwrap();
    let revision = manifest["dev-dependencies"]["camelid"]["rev"]
        .as_str()
        .expect("dev-dependencies.camelid.rev must pin one parity-oracle revision");
    assert!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "dev-dependencies.camelid.rev must be a full 40-character hexadecimal commit"
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

    // The identity of the engine that actually serves.
    //
    // `ENTERPRISE_ENGINE_PIN` above names the parity *oracle* — the revision this
    // engine is checked against — not the engine. Since the in-tree cutover
    // nothing published identified which build of `engine-core` produced a
    // token, so two Enterprise builds with different forward passes were
    // indistinguishable from outside. This closes that.
    //
    // A digest over `engine-core`'s sources rather than a git revision, on
    // purpose: it is correct in a tarball, in a dirty tree, and in CI, and it
    // moves when and only when the engine's source moves. A git rev would be
    // wrong in the first two cases and would move for changes that cannot
    // affect a token.
    let engine_src = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../engine-core/src")
        .canonicalize()
        .expect("engine-core/src must exist beside the server crate");
    let mut sources: Vec<std::path::PathBuf> = Vec::new();
    let mut stack = vec![engine_src.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("engine-core/src is readable") {
            let path = entry.expect("a readable directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                sources.push(path);
            }
        }
    }
    // Sorted, so the digest is a property of the tree and not of readdir order.
    sources.sort();
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    for source in &sources {
        let relative = source
            .strip_prefix(&engine_src)
            .expect("every source sits under engine-core/src");
        sha2::Digest::update(&mut hasher, relative.to_string_lossy().as_bytes());
        sha2::Digest::update(&mut hasher, b"\0");
        sha2::Digest::update(&mut hasher, std::fs::read(source).expect("a readable source file"));
        println!("cargo:rerun-if-changed={}", source.display());
    }
    println!("cargo:rerun-if-changed={}", engine_src.display());
    let engine_digest = format!("{:x}", sha2::Digest::finalize(hasher));
    println!("cargo:rustc-env=ENTERPRISE_ENGINE_DIGEST={engine_digest}");
}
