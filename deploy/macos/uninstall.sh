#!/bin/bash
#
# Remove the deterministic-lane daemon, and the gateway unit if it is installed.
#
#   sudo deploy/macos/uninstall.sh
#
# Removes the units, the rotation stanza and the binaries. Deliberately removes
# NO data: serving receipts are the audit trail this product exists to produce,
# and an uninstaller that deletes evidence as a side effect of tidying up is one
# nobody can safely run twice. The paths that survive are printed at the end,
# with the commands to remove them by hand.
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

die() { printf 'uninstall: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }

[[ "$(uname -s)" == "Darwin" ]] || die "this uninstaller is for macOS"
[[ "${EUID}" -eq 0 ]] || die "run with sudo"

# Gateway first, replica second -- the drain order. Stopping the replica while
# the gateway is still up turns every straggler into a client-visible 502
# instead of a refused connection.
#
# `bootout` sends SIGTERM, which both processes handle by refusing new
# connections and waiting for in-flight work. If requests are still arriving
# this can sit here for a long time: deregister at whatever fronts this box and
# poll the queue to zero first (README.md), or this is where you find out you
# did not.
for label in "${GATEWAY_LABEL}" "${LABEL}"; do
    if launchctl print "system/${label}" >/dev/null 2>&1; then
        note "stopping system/${label} -- waiting for in-flight work to finish"
        launchctl bootout "system/${label}" || true
    fi
done

for path in "${GATEWAY_PLIST}" "${PLIST}" "${ROTATE}" "${GATEWAY_BIN}" "${BIN}"; do
    if [[ -e "${path}" ]]; then
        rm -f "${path}"
        note "removed ${path}"
    fi
done

cat <<EOF

Removed the units, the log rotation and the binaries.

LEFT IN PLACE, on purpose:
    ${VAR}/receipts     serving receipts -- audit evidence, review before deleting
    ${VAR}/models       the GGUF this replica served
    ${LOGDIR}           launchd stdout/stderr
    account ${ACCOUNT}  owns the files above

To remove those as well, once you are sure the receipts are archived:
    sudo rm -rf ${VAR} ${LOGDIR}
    sudo dscl . -delete /Users/${ACCOUNT}
    sudo dscl . -delete /Groups/${ACCOUNT}

(Those dscl paths are DirectoryService node paths, not filesystem paths.)
EOF
