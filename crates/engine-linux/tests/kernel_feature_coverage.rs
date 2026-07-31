//! The reported feature vocabulary covers every kernel gate the pinned engine
//! takes by default, and claims nothing beyond what the fixture declares.
//!
//! `simd` is published on every response and is the field a client compares to
//! answer "did these two replicas route through the same kernels?". That
//! question is only answerable if the reported vocabulary covers what the
//! engine actually branches on. A feature the engine gates a kernel on and this
//! crate does not report gives two hosts the identical `simd=` string while one
//! of them takes a different code path.
//!
//! None of this is visible when reading the probe, which is why it is a test
//! rather than a comment: the gap it closes is the *absence* of a name, and
//! absences do not draw attention in review. The lane's canonical vector is
//! defended the same way, against a fixture pinned at the same engine revision.
//!
//! WHAT THIS PROVES, AND WHAT IT DOES NOT. It proves a set relation between two
//! compile-time constants and a checked-in table. That is deliberately all it
//! needs, because it is the entire failure mode: this crate's code is Linux
//! code that the macOS runner cannot execute and the x86-64 runner cannot
//! execute the aarch64 half of, so an execution-dependent check would be a test
//! that passes by not running. The vocabularies are plain `const`s with no
//! `#[cfg]` on them, so both architectures' lists are asserted on every runner,
//! on a host that may have none of the features named.
//!
//! It does not prove the fixture describes the engine. Nothing here reads the
//! engine; the table is a sweep of the pin, transcribed by hand, and its
//! `site` column exists so a reviewer can check a row in one step. It also does
//! not prove that a host with a feature takes the kernel — only that if the
//! engine routes on it, the identity can tell that host apart.

use engine_linux::{AARCH64_REPORTED_FEATURES, X86_64_REPORTED_FEATURES};

const FIXTURE: &str = include_str!("fixtures/engine-kernel-features.tsv");

/// Architectures this crate declares a vocabulary for. A fixture row naming
/// anything else is a typo that would otherwise be silently unchecked.
fn vocabularies() -> [(&'static str, &'static [&'static str]); 2] {
    [("x86_64", X86_64_REPORTED_FEATURES), ("aarch64", AARCH64_REPORTED_FEATURES)]
}

struct Row {
    arch: &'static str,
    feature: &'static str,
    routing: &'static str,
    published: &'static str,
    site: &'static str,
}

fn rows() -> Vec<Row> {
    FIXTURE
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split('\t');
            let mut next = |column| {
                fields.next().unwrap_or_else(|| panic!("fixture row is missing {column}: {line}"))
            };
            Row {
                arch: next("arch"),
                feature: next("feature"),
                routing: next("routing"),
                published: next("published"),
                site: next("site"),
            }
        })
        .collect()
}

/// The safety property. Everything else in this file exists to keep this one
/// from passing vacuously or being edited around.
#[test]
fn every_default_reachable_kernel_gate_is_published() {
    for (arch, reported) in vocabularies() {
        for row in rows().iter().filter(|row| row.arch == arch && row.published == "required") {
            assert!(
                reported.contains(&row.feature),
                "the pinned engine selects an inference kernel on {arch}/{} ({}), and this crate \
                 does not report it. Two hosts differing only in that feature publish the \
                 identical simd= string while routing through different kernels. Add it to the \
                 vocabulary; or, if the pin moved the gate behind an environment key the startup \
                 scan refuses, change the row's routing column and say so in the fixture.",
                row.feature,
                row.site
            );
        }
    }
}

/// The other direction, and not a cosmetic one: a name published without a
/// gate widens the identity with a distinction that changes no routing, so two
/// replicas that agree about everything the engine acts on can still publish
/// different strings. Rows tagged `optional` are the deliberate exceptions and
/// carry their reason in the fixture; `no` rows are names that must stay out.
#[test]
fn nothing_is_published_that_the_fixture_does_not_declare() {
    let rows = rows();
    for (arch, reported) in vocabularies() {
        for feature in reported {
            let row = rows.iter().find(|row| row.arch == arch && row.feature == *feature);
            let Some(row) = row else {
                panic!(
                    "{arch} reports {feature}, which is not in the kernel-feature fixture at all. \
                     Either the pin dropped that detection — in which case the fixture is stale — \
                     or the name was added to the published identity for a reason that belongs in \
                     the fixture, where the next person to read the vocabulary will find it."
                );
            };
            assert_ne!(
                row.published, "no",
                "{arch} reports {feature}, which the fixture marks as one no probe should \
                 publish ({}). The engine does not select a kernel on it, so publishing it adds \
                 a name to the identity that no client can act on.",
                row.site
            );
        }
    }
}

/// Guards the two assertions above against the edit that would defeat them:
/// demoting a routed feature to `optional` so the coverage test stops asking
/// for it. The two columns are one fact recorded twice, and this is the check
/// that they stay one fact.
#[test]
fn required_is_exactly_the_default_reachable_class() {
    for row in rows() {
        let routes_by_default = row.routing == "routes_by_default";
        let required = row.published == "required";
        assert_eq!(
            routes_by_default,
            required,
            "{}/{} is routing={} published={} ({}). A feature the engine routes on by default is \
             required and nothing else is: a row that routes but is not required would let the \
             coverage assertion pass while the identity under-reports, and a row that is required \
             but does not route asks every probe to publish a name for no reason.",
            row.arch,
            row.feature,
            row.routing,
            row.published,
            row.site
        );
    }
}

/// Guards the guards: every assertion above is vacuously true against an empty,
/// truncated, or misspelled table.
#[test]
fn the_fixture_is_intact() {
    let rows = rows();
    assert!(
        rows.len() >= 18,
        "the kernel-feature fixture parsed {} rows; the sweep at the pin recorded 18, so the file \
         or the parser is wrong and every assertion in this file is weaker than it reads",
        rows.len()
    );

    let mut seen = std::collections::BTreeSet::new();
    for row in &rows {
        assert!(
            vocabularies().iter().any(|(arch, _)| *arch == row.arch),
            "fixture names architecture {:?}, which this crate has no vocabulary for, so that \
             row constrains nothing",
            row.arch
        );
        assert!(
            matches!(
                row.routing,
                "routes_by_default" | "routes_only_under_a_refused_key" | "reporting_only"
            ),
            "{}/{} has routing={:?}, which is not one of the three classes the header defines; a \
             misspelling here silently exempts the row from the coverage assertion",
            row.arch,
            row.feature,
            row.routing
        );
        assert!(
            matches!(row.published, "required" | "optional" | "no"),
            "{}/{} has published={:?}, which is not one of the three the header defines",
            row.arch,
            row.feature,
            row.published
        );
        assert!(
            row.site.contains(".rs:"),
            "{}/{} has no source location to check the claim against",
            row.arch,
            row.feature
        );
        assert!(
            !row.site.starts_with('/'),
            "{}/{} cites an absolute path; sites are engine-relative so the table means the same \
             thing in every checkout",
            row.arch,
            row.feature
        );
        assert!(
            seen.insert((row.arch, row.feature)),
            "{}/{} appears twice; two rows for one feature can disagree, and the assertions above \
             would then depend on which one they happened to read",
            row.arch,
            row.feature
        );
    }

    for (arch, _) in vocabularies() {
        assert!(
            rows.iter().any(|row| row.arch == arch && row.published == "required"),
            "no required {arch} rows, so that architecture's vocabulary is effectively unchecked"
        );
    }
}

/// `probe` sorts the detected set but does not deduplicate it, so a duplicate
/// in a vocabulary would reach the wire as a duplicate in `simd=`. Sorting the
/// vocabulary itself is not needed for that — `probe` sorts — but an unsorted
/// list is how a name gets added twice without anyone noticing.
#[test]
fn each_vocabulary_is_sorted_and_unique() {
    for (arch, reported) in vocabularies() {
        let mut expected = reported.to_vec();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(
            reported, &expected[..],
            "{arch} vocabulary is not sorted and unique; simd= is compared for string equality \
             between replicas, so the rendering has to be a function of the feature set alone"
        );
    }
}
