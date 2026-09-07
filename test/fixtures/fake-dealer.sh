#!/bin/bash
# Stands in for shielded-dealer in tests: writes 1000 random bytes per requested range,
# for --out/--ranges or for every line of a --jobs file ("seed seed_id pk out-template ranges").
out=""; ranges=""; jobs=""
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift;; --ranges) ranges="$2"; shift;; --jobs) jobs="$2"; shift;; esac; shift; done
mint() {  # $1 = out template, $2 = ranges
  local IFS=,
  for r in $2; do
    local i0=${r%%:*} c=${r##*:}
    local f="${1//\{index0\}/$i0}"; f="${f//\{count\}/$c}"
    head -c 1000 /dev/urandom > "$f"; echo "minted $f"
  done
}
if [ -n "$jobs" ]; then
  while read -r seed sid pk tmpl rng; do [ -n "$tmpl" ] && mint "$tmpl" "$rng"; done < "$jobs"
else
  mint "$out" "$ranges"
fi
