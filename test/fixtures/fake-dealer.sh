#!/bin/bash
# Stands in for shielded-dealer in tests: writes 1000 random bytes per requested range.
out=""; ranges=""
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift;; --ranges) ranges="$2"; shift;; esac; shift; done
IFS=,
for r in $ranges; do
  i0=${r%%:*}; c=${r##*:}
  f="${out//\{index0\}/$i0}"; f="${f//\{count\}/$c}"
  head -c 1000 /dev/urandom > "$f"; echo "minted $f"
done
