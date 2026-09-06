#!/usr/bin/env bash
# deploy.sh - push the relay daemons + systemd units to their boxes and
# restart them. Paths are relative to relay/ however this script is invoked.
#
# HOSTS (ssh aliases; Host blocks in ~/.ssh/config, CI writes equivalents):
#   $RELAY_HOSTS - every data-plane relay this operator runs (default
#               "nan-relay"; CI passes the full list). Each gets the same
#               payload and decides for itself which units to run, from its own
#               env files - so a box that owns no /64 quietly skips tcp6/udp
#               instead of failing the deploy.
#   nan-relay - the TCP (SNI) + UDP relays. relay.js binds the whole 1-49999
#               public port range there, so the API relay CANNOT share this
#               box (its port 8100 sits inside that range). The box pins
#               net.ipv4.ip_local_port_range to 58000-65535 so its own outbound
#               ephemerals never collide with the listener range.
#   nan       - the API relay (api.enclave.host: the box's Caddy fronts :8100).
#
# Host layout (see README): /opt/nan-relay/ holds the daemons and their
# node_modules; units live in /etc/systemd/system; env files under
# /etc/nan-relay/ are host state and are NOT touched here.
set -euo pipefail
cd "$(dirname "$0")"

# DEPENDENCIES: `npm ci` against a SHIPPED package-lock.json, never `npm install`
# against package.json alone. Every dependency here is a caret range, so
# resolving them on the box means whatever the registry serves that morning —
# on the host that holds the Stripe webhook secret, the accounts store, the
# vault relayer key and the payment indexer. The Dockerfile and cli/install.sh
# both already say this in their own words; this path was the one that drifted.
# A registry failure aborts before any restart (set -e, && chaining): the
# running processes keep serving from their already-loaded module graph, and
# the next deploy repairs the tree.
# --- WHICH data-plane relays to ship to ---------------------------------------
# Every relay this operator runs, not a hardcoded one. The list was `nan-relay`
# alone for as long as there was one box, and the second (us-west, 2026-08-12)
# silently stopped receiving relay/** pushes the moment it existed - it carried
# the whole app zone on code from before it was added. A deploy that skips a
# live box is worse than one that fails: nothing says so.
#
# A LIST OF SSH ALIASES, not the on-chain relay registry, and deliberately so.
# Registration is permissionless - anyone may register a relay and most are not
# ours. "Deploy to every registered relay" would mean SSHing into strangers'
# machines, which cannot work and should not be attempted. The boxes we can
# deploy to are exactly the boxes we hold a key for, so the key list is the
# honest source.
#
# RELAY_HOSTS overrides (space- or comma-separated); CI passes the same names it
# writes Host blocks for. The default keeps the historical single box, so an
# operator who sets nothing deploys exactly as before.
RELAY_HOSTS="${RELAY_HOSTS:-nan-relay}"
RELAY_HOSTS=$(printf '%s' "$RELAY_HOSTS" | tr ',' ' ')
echo "== data-plane relays: $RELAY_HOSTS"

for RH in $RELAY_HOSTS; do
echo "== $RH: tcp (SNI) + tcp6 (dedicated-IP) + udp + egress relays"
# --- one-time port-range widening (2026-08-05): logical labels grew from
# 1-19999 to 1-49999 (supervisor parseFirewall / wasm_manager PORT_MAX_DECL /
# site validator moved together). The listener range now overlaps the kernel's
# default ephemeral ports (32768-60999), so the box pins its outbound
# ephemerals ABOVE the range BEFORE the wider bind — otherwise range binds race
# whatever outbound connections hold those ports and the skipped listeners stay
# dark until the next restart. Idempotent. The env edit is a surgical flip of
# the exact old value: any other RELAY_PORTS value is left alone (the grep
# below prints what the restart will actually bind). This is a deliberate
# exception to "env files are host state, never touched here", scoped to the
# one migration, like the unit-rename block below.
ssh "$RH" 'printf "net.ipv4.ip_local_port_range = 58000 65535\n" > /etc/sysctl.d/90-enclave-relay-ephemeral.conf \
  && sysctl -q -p /etc/sysctl.d/90-enclave-relay-ephemeral.conf \
  && sed -i "s/^RELAY_PORTS=1-19999$/RELAY_PORTS=1-49999/" /etc/nan-relay/tcp-relay.env \
  && grep -H "^RELAY_PORTS=" /etc/nan-relay/tcp-relay.env \
  && sysctl net.ipv4.ip_local_port_range'
# net-guard.mjs is a symlink to ../net-guard.mjs (the canonical SSRF classifier
# shared with the enclave's egress.js); scp follows it and ships the content.
# fleet.mjs is the shared fleet discovery (REGISTRY_ADDRESS / ENCLAVES) the
# tcp6/udp/egress relays use to follow an arbitrary, changing set of enclaves.
# relay-agent.mjs is how a relay box reports itself to the fleet - it dials the
# hub and answers /availability and the connection log at /t/<name>. It had NO
# deploy path at all: the unit lived in this repo but was only ever installed by
# hand, so the one daemon whose job is to tell the fleet what this box is could
# not be updated by a deploy. us-west answered /availability from months-old
# code while 404ing routes that had shipped.
scp relay.js tcp6-relay.js udp-relay.js egress-relay.js dns-relay.js relay-agent.mjs fleet.mjs net-guard.mjs connlog.mjs boxhost.js package.json package-lock.json "$RH":/opt/nan-relay/
scp systemd/enclave-tcp-relay.service systemd/enclave-tcp6-relay.service systemd/enclave-udp-relay.service systemd/enclave-egress-relay.service systemd/enclave-dns.service systemd/enclave-relay-agent.service "$RH":/etc/systemd/system/
# WHICH UNITS a box runs is the box's own answer, read off its env files rather
# than assumed. Only the SNI relay is universal: tcp6 and udp bind per-deployment
# addresses out of a routed /64, so a relay that does not own one (us-west) would
# fail EADDRNOTAVAIL and take the whole deploy down with it - for daemons it was
# never meant to run. Same shape as the egress/dns gates below, which have always
# worked this way.
# The egress relay only runs once /etc/nan-relay/egress-relay.env exists
# (REGISTRY_ADDRESS or ENCLAVES + EGRESS_RELAY_TOKEN + EGRESS_PREFIX=<same
# /64>). Until then its restart is a no-op failure; enable it explicitly when
# the operator adds the env.
# One-time migration from the pre-rename nan-* unit names: the old unit must
# be gone before the enclave-* one starts, or the two race for the same ports.
ssh "$RH" 'for u in nan-tcp-relay nan-tcp6-relay nan-udp-relay nan-egress-relay; do \
    if [ -f /etc/systemd/system/$u.service ]; then \
      systemctl disable --now $u || true; rm /etc/systemd/system/$u.service; fi; done \
  && cd /opt/nan-relay && npm ci --omit=dev --no-audit --no-fund \
  && systemctl daemon-reload \
  && UNITS=enclave-tcp-relay \
  && for pair in enclave-tcp6-relay:tcp6-relay.env enclave-udp-relay:udp-relay.env; do \
       u=${pair%%:*}; f=${pair##*:}; \
       if [ -f "/etc/nan-relay/$f" ]; then UNITS="$UNITS $u"; \
       else echo "$u: no /etc/nan-relay/$f here - skipped"; fi; done \
  && systemctl enable $UNITS \
  && systemctl restart $UNITS \
  && sleep 4 \
  && if systemctl is-active --quiet $UNITS; then echo "data plane active: $UNITS"; \
     else echo "a data-plane relay FAILED to stay up after restart (crash loop?):"; \
          systemctl is-active $UNITS || true; \
          journalctl $(for u in $UNITS; do printf " -u %s" "$u"; done) -n 25 --no-pager; exit 1; fi \
  && if [ -f /etc/nan-relay/egress-relay.env ]; then \
       systemctl enable --now enclave-egress-relay && systemctl restart enclave-egress-relay \
       && systemctl is-active enclave-egress-relay; \
     else echo "enclave-egress-relay: no /etc/nan-relay/egress-relay.env yet — skipped"; fi \
  && if [ -f /etc/nan-relay/dns.env ]; then \
       systemctl enable --now enclave-dns && systemctl restart enclave-dns \
       && systemctl is-active enclave-dns; \
     else echo "enclave-dns: no /etc/nan-relay/dns.env yet — skipped (authoritative DNS for app./ip. zones)"; fi \
  && if [ -f /etc/nan-relay/relay-agent.env ]; then \
       systemctl enable --now enclave-relay-agent && systemctl restart enclave-relay-agent \
       && systemctl is-active enclave-relay-agent; \
     else echo "enclave-relay-agent: no /etc/nan-relay/relay-agent.env yet — skipped (this box does not report itself)"; fi'
done


# --- secret-bearing env files: check, never touch ---------------------------
# /etc/nan-relay/*.env hold real secrets — PROVISIONER_PRIVATE_KEY is a funded
# Base key that moves USDC, alongside STRIPE_SECRET_KEY, SECRETS_KEY and
# UPLOAD_KEY. systemd reads them as root before dropping to the DynamicUser, so
# nothing needs them group- or world-readable. Nothing in this repo has ever
# checked, and a key you cannot rule out as leaked is a key you have to rotate.
# Reported, not modified: this script promises not to touch host env state, and
# a loud line the operator acts on beats a silent chmod they never see.
check_env_perms() {
  ssh "$1" 'for f in /etc/nan-relay/*.env; do [ -e "$f" ] || continue;
    m=$(stat -c %a "$f"); o=$(stat -c %U "$f");
    case "$m" in *[1-7]|*[1-7]?) echo "  !! $f is mode $m (owner $o) — readable beyond its owner; run: sudo chmod 600 $f" ;;
                 *) echo "  ok $f mode $m ($o)" ;; esac; done' || true
}
echo "== env-file permissions (secrets live here)"
check_env_perms nan-relay
check_env_perms nan

echo "== api relay (site box)"
# api-relay.js imports ./fleet.mjs (shared discovery: registry read + TRUSTED_OPERATORS
# filter + on-chain runner routing), ./net-guard.mjs (SSRF classifier for discovered
# origins), ./tunnel.js (fleet tunnel for CGNAT self-hosted enclaves) AND ./mcp.js
# (the MCP coding-agent endpoint, mcp.enclave.host); fleet.mjs imports ./net-guard.mjs
# too. ALL of them MUST ship alongside or the service crash-loops with ERR_MODULE_NOT_FOUND.
# auth/billing modules (account sessions, orders, Stripe webhook, PaymentRouter
# indexer, OFAC screen, provisioner) ship alongside; they self-disable without
# StateDirectory/env, so shipping them is always safe. npm ci below installs
# their deps (@simplewebauthn/server, jose) from the SHIPPED lockfile.
scp api-relay.js mcp.js auth.js sso.js billing.js indexer.js ofac.js provisioner.js vaultsvc.js secrets.js fleet-auth.js certs.js domains.js store.js fleet.mjs net-guard.mjs tunnel.js snp-verify.mjs avf-verify.mjs pads.mjs boxhost.js package.json package-lock.json nan:/opt/nan-relay/
scp systemd/enclave-api-relay.service nan:/etc/systemd/system/
ssh nan 'if [ -f /etc/systemd/system/nan-api-relay.service ]; then \
    systemctl disable --now nan-api-relay || true; rm /etc/systemd/system/nan-api-relay.service; fi \
  && cd /opt/nan-relay && npm ci --omit=dev --no-audit --no-fund \
  && systemctl daemon-reload \
  && systemctl enable enclave-api-relay \
  && systemctl restart enclave-api-relay \
  && sleep 4 \
  && if systemctl is-active --quiet enclave-api-relay; then echo "enclave-api-relay: active"; \
     else echo "enclave-api-relay FAILED to stay up after restart (crash loop?) — last logs:"; \
          journalctl -u enclave-api-relay -n 25 --no-pager; exit 1; fi'
