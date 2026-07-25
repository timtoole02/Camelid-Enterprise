#!/bin/bash
#
# Install the deterministic-lane replica as a launchd system daemon, and
# optionally the transparent gateway alongside it.
#
#   sudo deploy/macos/install.sh [--with-gateway] [path/to/camelid-enterprise]
#
# Idempotent: re-running it upgrades the binaries and the units in place. It
# does not start the replica unless a model is already in place, because the
# replica fails closed on a missing model and a KeepAlive job that fails closed
# is a restart loop, not a diagnostic.
#
# --with-gateway is off by default on purpose. The gateway forwards to one fixed
# origin, so behind a load balancer a per-box gateway balances nothing and adds
# a second process whose shutdown window has to be sized to a generation. Use it
# on a Mac that serves clients directly, so the box has one entry point and the
# replica can stay on loopback. See README.md, "Does this box run the gateway?".
#
# Everything lands outside any repository checkout and outside the directories
# macOS asks a user's consent for -- a daemon has no session to answer a consent
# prompt on -- and every path is space-free because newsyslog's config format is
# whitespace-delimited.
set -euo pipefail

LABEL="com.camelid.enterprise"
GATEWAY_LABEL="com.camelid.enterprise.gateway"
PREFIX="/usr/local"
BIN="${PREFIX}/bin/camelid-enterprise"
GATEWAY_BIN="${PREFIX}/bin/camelid-enterprise-gateway"
VAR="${PREFIX}/var/camelid-enterprise"
LOGDIR="${PREFIX}/var/log/camelid-enterprise"
PLIST="/Library/LaunchDaemons/${LABEL}.plist"
GATEWAY_PLIST="/Library/LaunchDaemons/${GATEWAY_LABEL}.plist"
ROTATE="/etc/newsyslog.d/${LABEL}.conf"
ACCOUNT="_camelid"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SRC}/../.." && pwd)"

WITH_GATEWAY=0
SOURCE_BIN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-gateway) WITH_GATEWAY=1 ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) printf 'install: unknown option %s\n' "$1" >&2; exit 2 ;;
        *) SOURCE_BIN="$1" ;;
    esac
    shift
done
SOURCE_BIN="${SOURCE_BIN:-${ROOT}/target/release/camelid-enterprise}"
SOURCE_GATEWAY_BIN="$(dirname "${SOURCE_BIN}")/camelid-enterprise-gateway"

die() { printf 'install: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }

[[ "$(uname -s)" == "Darwin" ]] || die "this installer is for macOS"
[[ "${EUID}" -eq 0 ]] || die "run with sudo -- it creates a service account and writes to /Library"
[[ -x "${SOURCE_BIN}" ]] \
    || die "no binary at ${SOURCE_BIN}; run: cargo build --release --bin camelid-enterprise"
if [[ "${WITH_GATEWAY}" -eq 1 ]]; then
    [[ -x "${SOURCE_GATEWAY_BIN}" ]] \
        || die "no gateway binary at ${SOURCE_GATEWAY_BIN}; run: cargo build --release --bin camelid-enterprise-gateway"
fi

# Fail before touching anything if the artifacts are malformed or the binary
# does not run on this machine.
plutil -lint "${SRC}/${LABEL}.plist" >/dev/null || die "the replica plist does not parse"
"${SOURCE_BIN}" --version >/dev/null || die "${SOURCE_BIN} does not run on this machine"
if [[ "${WITH_GATEWAY}" -eq 1 ]]; then
    plutil -lint "${SRC}/${GATEWAY_LABEL}.plist" >/dev/null || die "the gateway plist does not parse"
    "${SOURCE_GATEWAY_BIN}" --version >/dev/null || die "${SOURCE_GATEWAY_BIN} does not run on this machine"
fi

# --- service account ---------------------------------------------------------
# A hidden role account in the reserved system id range. Not `nobody`: that
# account is shared by everything on the machine, so it grants no isolation.
#
# The dscl paths below (/Users/..., /Groups/...) are DirectoryService NODE
# paths, not filesystem paths. Nothing here reads or writes a home directory.
free_id() { # $1 = Users|Groups, $2 = UniqueID|PrimaryGroupID
    local used id
    used="$(dscl . -list "/$1" "$2" | awk '{print $2}')"
    for id in $(seq 200 400); do
        grep -qx "${id}" <<<"${used}" || { echo "${id}"; return; }
    done
    die "no free system id in 200-400 for /$1"
}

if ! dscl . -read "/Groups/${ACCOUNT}" >/dev/null 2>&1; then
    gid="$(free_id Groups PrimaryGroupID)"
    dscl . -create "/Groups/${ACCOUNT}"
    dscl . -create "/Groups/${ACCOUNT}" PrimaryGroupID "${gid}"
    dscl . -create "/Groups/${ACCOUNT}" RealName "Camelid Enterprise deterministic lane"
    dscl . -create "/Groups/${ACCOUNT}" Password "*"
    note "created group ${ACCOUNT} (gid ${gid})"
fi
gid="$(dscl . -read "/Groups/${ACCOUNT}" PrimaryGroupID | awk '{print $2}')"

if ! dscl . -read "/Users/${ACCOUNT}" >/dev/null 2>&1; then
    uid="$(free_id Users UniqueID)"
    dscl . -create "/Users/${ACCOUNT}"
    dscl . -create "/Users/${ACCOUNT}" UniqueID "${uid}"
    dscl . -create "/Users/${ACCOUNT}" PrimaryGroupID "${gid}"
    dscl . -create "/Users/${ACCOUNT}" RealName "Camelid Enterprise deterministic lane"
    dscl . -create "/Users/${ACCOUNT}" UserShell /usr/bin/false
    dscl . -create "/Users/${ACCOUNT}" NFSHomeDirectory /var/empty
    dscl . -create "/Users/${ACCOUNT}" Password "*"
    dscl . -create "/Users/${ACCOUNT}" IsHidden 1
    note "created account ${ACCOUNT} (uid ${uid}), no shell, no password"
fi

# --- layout ------------------------------------------------------------------
# The model is owned by root and only readable by the service account: a replica
# that can rewrite its own model can change what it serves without changing
# anything it publishes about itself.
#
# Shared parents are created only when absent and never re-chowned -- on an
# Intel Mac /usr/local/bin and /usr/local/var belong to Homebrew, and taking them
# for root would break every formula on the machine.
for shared in "${PREFIX}/bin" "${PREFIX}/var" "${PREFIX}/var/log"; do
    [[ -d "${shared}" ]] || install -d -o root -g wheel -m 0755 "${shared}"
done
install -d -o root -g "${ACCOUNT}" -m 0755 "${VAR}"
install -d -o root -g "${ACCOUNT}" -m 0755 "${VAR}/models"
install -d -o "${ACCOUNT}" -g "${ACCOUNT}" -m 0750 "${VAR}/receipts"
install -d -o "${ACCOUNT}" -g "${ACCOUNT}" -m 0750 "${LOGDIR}"

# launchd creates its stdio files on first spawn if they are absent, at whatever
# the job's umask yields. Pre-creating them makes the mode deterministic and
# equal to what the layout table in README.md promises: these logs name what the
# replica served, so they are group-readable at most. Existing files are left
# alone -- a mode an operator widened on purpose is not ours to reverse.
stdio_files=(lane.out.log lane.err.log)
[[ "${WITH_GATEWAY}" -eq 1 ]] && stdio_files+=(gateway.out.log gateway.err.log)
for stdio in "${stdio_files[@]}"; do
    [[ -e "${LOGDIR}/${stdio}" ]] \
        || install -o "${ACCOUNT}" -g "${ACCOUNT}" -m 0640 /dev/null "${LOGDIR}/${stdio}"
done

# --- stop, install, restart --------------------------------------------------
# Stop first if either is running: replacing a binary under a live process
# leaves it serving from an unlinked inode while the unit claims otherwise.
#
# Gateway before replica, the same order as a drain. Stopping the replica first
# leaves the gateway answering stragglers with a typed 502, so an upgrade
# produces client-visible errors instead of refused connections.
for label in "${GATEWAY_LABEL}" "${LABEL}"; do
    if launchctl print "system/${label}" >/dev/null 2>&1; then
        note "stopping system/${label} (waits for in-flight generations)"
        launchctl bootout "system/${label}" || true
    fi
done

install -o root -g wheel -m 0755 "${SOURCE_BIN}" "${BIN}"
install -o root -g wheel -m 0644 "${SRC}/${LABEL}.plist" "${PLIST}"
install -o root -g wheel -m 0644 "${SRC}/${LABEL}.conf" "${ROTATE}"
note "installed ${BIN}, ${PLIST}, ${ROTATE}"

if [[ "${WITH_GATEWAY}" -eq 1 ]]; then
    install -o root -g wheel -m 0755 "${SOURCE_GATEWAY_BIN}" "${GATEWAY_BIN}"
    install -o root -g wheel -m 0644 "${SRC}/${GATEWAY_LABEL}.plist" "${GATEWAY_PLIST}"
    note "installed ${GATEWAY_BIN}, ${GATEWAY_PLIST}"
elif [[ -e "${GATEWAY_PLIST}" ]]; then
    note "NOTE: ${GATEWAY_PLIST} is installed but --with-gateway was not passed."
    note "      It was stopped above and will not be restarted. Run uninstall.sh"
    note "      to remove it, or re-run with --with-gateway."
fi

model="$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments' "${PLIST}" 2>/dev/null \
    | awk '/^ *--model$/{getline; print $1}')"
if [[ -z "${model}" ]]; then
    note "could not read --model out of ${PLIST}; skipping the pre-start check"
elif [[ ! -f "${model}" ]]; then
    cat <<EOF

Installed, not started. The unit expects a model at:
    ${model}

Put a GGUF there (owned root, mode 0644, readable by ${ACCOUNT}) or edit
${PLIST}, then:
    sudo launchctl bootstrap system ${PLIST}

Review --threads in the unit before starting. It sizes this replica's worker
pool, the resolved width is published on every response, and it must be
identical on every replica in a pool.
EOF
    exit 0
fi

launchctl bootstrap system "${PLIST}"
note "bootstrapped system/${LABEL}"
if [[ "${WITH_GATEWAY}" -eq 1 ]]; then
    launchctl bootstrap system "${GATEWAY_PLIST}"
    note "bootstrapped system/${GATEWAY_LABEL}"
fi

cat <<EOF

Started. Listening is not readiness: the replica reads and hashes the model whole
BEFORE it binds its port -- a replica must never answer a request without being
able to say what produced it -- then binds, then loads the model, and serves
nothing until that load finishes.

    until curl -fsS --max-time 3 http://127.0.0.1:8181/v1/health \\
        | grep -q '"generation_ready":true'; do sleep 2; done

Then check what it published about itself:

    curl -sS -D - -o /dev/null http://127.0.0.1:8181/v1/health | grep -i '^x-camelid'

Logs: ${LOGDIR}/lane.err.log   Receipts: ${VAR}/receipts/serving.jsonl
Stop (drained): follow the drain sequence in deploy/macos/README.md
EOF
