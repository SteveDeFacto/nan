#!/usr/bin/env bash
# build.sh -- one APK carrying a payload for an AVF protected VM.
#
#   ./build.sh attest_probe         # the RKP attestation probe
#   ./build.sh anchor               # the anchor itself (core + simd + field)
#   ./build.sh sink                 # host-side vsock sink for --debug none runs
#   ./build.sh probe                # the complete trusted half + shielded-probe, static, for the phone
#   ./build.sh engine               # libggml-shielded.so + ggml-test + shielded-run for the phone (see build-ggml-arm64.sh)
#
# Produces out/<name>.apk, signed with keys/anchor.jks. Then, on the device:
#   vm create-idsig <apk> <idsig>
#   vm run-app --payload-binary-name lib<name>.so --protected [--debug none] ...
#
# The payload is a bionic .so with DT_NEEDED libvm_payload.so. That library
# exists only inside Microdroid, so we link against a STUB generated from
# AOSP's symbol map (libvm_payload.map.txt): empty functions with the right
# names, enough for the linker, never shipped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:-attest_probe}"
SDK="${ANDROID_HOME:-$HOME/Android/Sdk}"
NDK="$SDK/ndk/27.2.12479018"
BT="$SDK/build-tools/35.0.0"
API=35
CLANG="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${API}-clang"
HDR="${AVF_HDR:-$HERE}"          # $HDR/avfref/{vm_payload.h,libvm_payload.map.txt}: vendored from AOSP (avfref/NOTICE)
GG="$HERE/../../../wasm/ggml-shielded"
CORE="$HERE/../core"
OUT="$HERE/out"; STUB="$OUT/stub"; STAGE="$OUT/stage-$NAME"
mkdir -p "$STUB" "$STAGE/lib/arm64-v8a" "$STAGE/assets"

# The APK signing key is the payload's identity to a verifier (its SHA-512 is the
# attested authorityHash). It is generated locally and never committed; a real
# deployment replaces it with a key held in the platform certificate service.
KS="$HERE/keys/anchor.jks"
if [ ! -f "$KS" ]; then
  mkdir -p "$HERE/keys"
  keytool -genkeypair -keystore "$KS" -storepass anchor123 -keypass anchor123 -alias anchor \
    -keyalg RSA -keysize 2048 -validity 3650 -dname "CN=Enclave Anchor Spike, O=Enclave Host" >/dev/null 2>&1
  echo "generated spike signing key $KS"
fi

for t in "$CLANG" "$BT/aapt2" "$BT/apksigner" "$BT/zipalign" "$SDK/platforms/android-$API/android.jar"; do
  [ -e "$t" ] || { echo "missing: $t" >&2; exit 2; }
done

# --- 1. stub libvm_payload.so from the symbol map -------------------------
if [ ! -f "$STUB/libvm_payload.so" ]; then
  MAP="$HDR/avfref/libvm_payload.map.txt"
  { echo '/* generated: link-time stub for Microdroid libvm_payload.so */';
    grep -oE 'AVm[A-Za-z_]+' "$MAP" | sort -u | while read -r s; do echo "void $s(void) {}"; done; } > "$STUB/stub.c"
  "$CLANG" -shared -fPIC -o "$STUB/libvm_payload.so" -Wl,-soname,libvm_payload.so "$STUB/stub.c"
  echo "stub: $(grep -c '^void' "$STUB/stub.c") symbols"
fi

# --- 2. the payload .so -----------------------------------------------------
CFLAGS=(-O2 -fPIC -Wall -march=armv8.2-a+dotprod -I"$HDR/avfref" -I"$CORE" -I"$GG")
case "$NAME" in
  sink)  # the host side of a --debug none VM's vsock report channel (static, runs from adb shell)
         "$CLANG" -O2 -static -Wall -o "$OUT/vsock-sink" "$HERE/host/vsock-sink.c"
         echo "sink: $OUT/vsock-sink ($(stat -c %s "$OUT/vsock-sink") bytes)"; exit 0 ;;
  probe)  # the COMPLETE trusted half (shielded-tee.c, not the anchor subset) + its probe, for the phone
         # itself: proves the file builds on bionic/aarch64 and that the NEON SDOT table agrees with
         # the generic one on this silicon (sh_simd_get runs simd_agree and prints the loser).
         PF=(-O2 -Wall -march=armv8.2-a+dotprod -I"$GG")
         "$CLANG" "${PF[@]}" -O3 -DSH_SIMD_NEON -c "$GG/shielded-simd.c" -o "$OUT/simd-neon.o"
         "$CLANG" "${PF[@]}" -O3 -c "$GG/shielded-simd.c" -o "$OUT/simd-generic.o"
         "$CLANG" "${PF[@]}" -ffp-contract=off -c "$GG/shielded-field.c" -o "$OUT/field.o"
         "$CLANG" "${PF[@]}" -c "$GG/shielded-wire.c" -o "$OUT/wire.o"
         "$CLANG" "${PF[@]}" -c "$GG/shielded-tee.c" -o "$OUT/tee.o"
         "$CLANG" "${PF[@]}" -c "$GG/shielded-pads.c" -o "$OUT/pads.o"      # dealt pads (shielded/dealer/PLAN.md)
         "$CLANG" "${PF[@]}" -w -c "$GG/tweetnacl.c" -o "$OUT/nacl.o"
         "$CLANG" "${PF[@]}" -O3 -w -c "$GG/poly1305-donna.c" -o "$OUT/poly.o"
         "$CLANG" "${PF[@]}" -static -o "$OUT/shielded-probe" "$GG/shielded-probe.c" "$OUT/tee.o" "$OUT/pads.o" "$OUT/nacl.o" "$OUT/poly.o" "$OUT/field.o" "$OUT/wire.o" "$OUT/simd-neon.o" "$OUT/simd-generic.o" -lm
         printf '#include "shielded-simd.h"\n#include <stdio.h>\nint main(void){printf("simd=%%s\\n", sh_simd_get()->name);return 0;}\n' > "$OUT/simd-check.c"
         "$CLANG" "${PF[@]}" -static -o "$OUT/simd-check" "$OUT/simd-check.c" "$OUT/tee.o" "$OUT/pads.o" "$OUT/nacl.o" "$OUT/poly.o" "$OUT/field.o" "$OUT/wire.o" "$OUT/simd-neon.o" "$OUT/simd-generic.o" -lm
         echo "probe: $OUT/shielded-probe ($(stat -c %s "$OUT/shielded-probe") bytes), simd-check"; exit 0 ;;
  engine)  # the COMPLETE engine for the phone, normal world: libggml-shielded.so (the backend module),
           # ggml-test and shielded-run, against the arm64 llama.cpp from build-ggml-arm64.sh.
           #   GGML_ARM64=<prefix dir>   default out/ggml-arm64-work/prefix
           # Run on the device with LD_LIBRARY_PATH=<dir> SHIELDED_SO=<dir>/libggml-shielded.so
           #   GGML_CPU_SO=<dir>/libggml-cpu.so SHIELDED_HOST/PORT/CALIB (REPORT.md section 12).
           GA="${GGML_ARM64:-$HERE/out/ggml-arm64-work/prefix}"; LSRC="$(dirname "$GA")/llama.cpp"
           [ -d "$GA/lib" ] || { echo "no arm64 llama.cpp at $GA; run build-ggml-arm64.sh first" >&2; exit 2; }
           CXX="${CLANG}++"; INC=(-I"$LSRC/include" -I"$LSRC/ggml/include" -I"$LSRC/ggml/src"); E="$OUT/engine"; mkdir -p "$E"
           PF=(-O2 -fPIC -march=armv8.2-a+dotprod -I"$GG")
           "$CLANG" "${PF[@]}" -O3 -DSH_SIMD_NEON -c "$GG/shielded-simd.c" -o "$E/simd-neon.o"
           "$CLANG" "${PF[@]}" -O3 -c "$GG/shielded-simd.c" -o "$E/simd-generic.o"
           "$CLANG" "${PF[@]}" -ffp-contract=off -c "$GG/shielded-field.c" -o "$E/field.o"
           "$CLANG" "${PF[@]}" -c "$GG/shielded-wire.c" -o "$E/wire.o"
           "$CLANG" "${PF[@]}" -c "$GG/shielded-tee.c" -o "$E/tee.o"
           "$CLANG" "${PF[@]}" -c "$GG/shielded-pads.c" -o "$E/pads.o"      # dealt pads (shielded/dealer/PLAN.md)
           "$CLANG" "${PF[@]}" -w -c "$GG/tweetnacl.c" -o "$E/nacl.o"
           "$CLANG" "${PF[@]}" -O3 -w -c "$GG/poly1305-donna.c" -o "$E/poly.o"
           CORE=("$E/tee.o" "$E/pads.o" "$E/nacl.o" "$E/poly.o" "$E/field.o" "$E/wire.o" "$E/simd-neon.o" "$E/simd-generic.o")
           "$CXX" -O2 -std=c++17 -fPIC -march=armv8.2-a+dotprod -DGGML_MAX_NAME=128 -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED "${INC[@]}" -I"$GG" -c "$GG/ggml-shielded.cpp" -o "$E/ggml-shielded-dl.o"
           # bionic does not resolve a dlopened module's symbols against the executable's other libraries: link libggml too
           "$CXX" -shared -o "$E/libggml-shielded.so" "$E/ggml-shielded-dl.o" "${CORE[@]}" -L"$GA/lib" -lggml -lggml-base -lm
           "$CXX" -O2 -std=c++17 -fPIC -march=armv8.2-a+dotprod -DGGML_MAX_NAME=128 "${INC[@]}" -I"$GG" -c "$GG/ggml-shielded.cpp" -o "$E/ggml-shielded.o"
           "$CXX" -O2 -std=c++17 -march=armv8.2-a+dotprod -DGGML_MAX_NAME=128 "${INC[@]}" -I"$GG" -o "$E/ggml-test" "$GG/ggml-test.cpp" "$E/ggml-shielded.o" "${CORE[@]}" -L"$GA/lib" -lggml -lggml-base -lggml-cpu -lm
           "$CXX" -O2 -std=c++17 -march=armv8.2-a+dotprod -DGGML_MAX_NAME=128 "${INC[@]}" -I"$GG" -o "$E/shielded-run" "$GG/shielded-run.cpp" -L"$GA/lib" -lllama -lggml -lggml-base -ldl -lm
           cp "$GA"/lib/libllama.so "$GA"/lib/libggml.so "$GA"/lib/libggml-base.so "$GA"/lib/libggml-cpu.so "$GA"/lib/libc++_shared.so "$GG/test.calib" "$E/"
           echo "engine: $E (push the directory to the phone)"; ls "$E" | grep -vE '\.o$' | tr '\n' ' '; echo; exit 0 ;;
  engine-pvm)  # the engine FOR THE VM: the shielded module with the fd-adopting hook, and libengine.so
           #   (shielded-run's flow, model from a memfd, worker fd adopted). Both are dlopened by the
           #   bootstrap payload from the APK, so libengine.so may carry ordinary DT_NEEDED on llama/ggml.
           GA="${GGML_ARM64:-$HERE/out/ggml-arm64-work/prefix}"; LSRC="$(dirname "$GA")/llama.cpp"
           [ -d "$GA/lib" ] || { echo "no arm64 llama.cpp at $GA; run build-ggml-arm64.sh first" >&2; exit 2; }
           CXX="${CLANG}++"; INC=(-I"$LSRC/include" -I"$LSRC/ggml/include" -I"$LSRC/ggml/src"); E="$OUT/engine-pvm"; mkdir -p "$E"
           PF=(-O2 -fPIC -march=armv8.2-a+dotprod -I"$GG" -I"$HERE/../harness")
           "$CLANG" "${PF[@]}" -O3 -DSH_SIMD_NEON -c "$GG/shielded-simd.c" -o "$E/simd-neon.o"
           "$CLANG" "${PF[@]}" -O3 -c "$GG/shielded-simd.c" -o "$E/simd-generic.o"
           "$CLANG" "${PF[@]}" -ffp-contract=off -c "$GG/shielded-field.c" -o "$E/field.o"
           "$CLANG" "${PF[@]}" -c "$HERE/../harness/wire-fd.c" -o "$E/wire-fd.o"                       # shielded-wire.c + sh_pipe_open_fd + the hook
           "$CLANG" "${PF[@]}" -Dsh_pipe_open=sh_pipe_open_hook -c "$GG/shielded-tee.c" -o "$E/tee.o"    # the trusted half dials through the hook
           "$CLANG" "${PF[@]}" -c "$GG/shielded-pads.c" -o "$E/pads.o"      # dealt pads (shielded/dealer/PLAN.md)
           "$CLANG" "${PF[@]}" -w -c "$GG/tweetnacl.c" -o "$E/nacl.o"
           "$CLANG" "${PF[@]}" -O3 -w -c "$GG/poly1305-donna.c" -o "$E/poly.o"
           "$CXX" -O2 -std=c++17 -fPIC -march=armv8.2-a+dotprod -DGGML_MAX_NAME=128 -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED "${INC[@]}" -I"$GG" -c "$GG/ggml-shielded.cpp" -o "$E/ggml-shielded-dl.o"
           "$CXX" -shared -o "$E/libggml-shielded.so" "$E/ggml-shielded-dl.o" "$E/tee.o" "$E/pads.o" "$E/nacl.o" "$E/poly.o" "$E/field.o" "$E/wire-fd.o" "$E/simd-neon.o" "$E/simd-generic.o" -L"$GA/lib" -lggml -lggml-base -lm -Wl,-soname,libggml-shielded.so
           "$CXX" -O2 -std=c++17 -fPIC -march=armv8.2-a+dotprod -DGGML_MAX_NAME=128 "${INC[@]}" -shared -o "$E/libengine.so" "$HERE/payload/engine.cpp" "$E/pads.o" "$E/nacl.o" "$E/poly.o" -L"$GA/lib" -lllama -lggml -lggml-base -llog -ldl -Wl,-soname,libengine.so
           "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm" -D "$E/libggml-shielded.so" | grep -E ' T (sh_pipe_adopt_fd|sh_pipe_open_hook|ggml_backend_shielded_stats)$' | sed 's/^/  /'
           echo "engine-pvm: $E/libggml-shielded.so ($(stat -c %s "$E/libggml-shielded.so") B), libengine.so ($(stat -c %s "$E/libengine.so") B)"; exit 0 ;;
  attest_probe) SRCS=("$HERE/payload/attest_probe.c") ;;
  pvm_probe)    SRCS=("$HERE/payload/pvm_probe.c"); EXTRA_LIBS=("$HOME/Android/Sdk/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "${GGML_ARM64:-$HERE/out/ggml-arm64-work/prefix}/lib/libggml-base.so" "${GGML_ARM64:-$HERE/out/ggml-arm64-work/prefix}/lib/libggml.so") ;;
  anchor)       # the anchor + the harness's worker client over an fd (wire-fd.c wraps the shipped shielded-wire.c).
                # shielded-simd.c is built twice, generic and -DSH_SIMD_NEON; the core's refill is pointed at SDOT.
                "$CLANG" -O3 -fPIC -march=armv8.2-a+dotprod -DSH_SIMD_NEON -I"$GG" -c "$GG/shielded-simd.c" -o "$OUT/simd-neon-pic.o"
                SRCS=("$HERE/payload/anchor_payload.c" "$CORE/anchor-core.c" "$GG/shielded-simd.c" "$GG/shielded-field.c"
                      "$HERE/../harness/worker-client.c" "$HERE/../harness/wire-fd.c" "$GG/shielded-pads.c" "$GG/poly1305-donna.c"
                      "$HERE/payload/third_party/tweetnacl.c" "$OUT/simd-neon-pic.o")
                CFLAGS+=(-ffp-contract=off -I"$HERE/../harness" -DAN_REFILL=sh_simd_neon_refill)
                # the engine rides along when it has been built (build.sh engine-pvm): six libraries + the calibration
                GA="${GGML_ARM64:-$HERE/out/ggml-arm64-work/prefix}"
                if [ -f "$OUT/engine-pvm/libengine.so" ]; then
                  EXTRA_LIBS=("$GA/lib/libc++_shared.so" "$GA/lib/libggml-base.so" "$GA/lib/libggml.so" "$GA/lib/libggml-cpu.so" "$GA/lib/libllama.so" "$OUT/engine-pvm/libggml-shielded.so" "$OUT/engine-pvm/libengine.so")
                  EXTRA_ASSETS=("${ANCHOR_CALIB:-$HERE/../../../metal/shielded-overlay/calib/qwen3.5-0.8b-mtp-gguf.calib}")
                  echo "engine: bundling ${#EXTRA_LIBS[@]} libraries + $(basename "${EXTRA_ASSETS[0]}") as assets/model.calib"
                fi ;;
  *) echo "unknown payload $NAME" >&2; exit 2 ;;
esac
rm -f "$STAGE"/lib/arm64-v8a/*.so
"$CLANG" "${CFLAGS[@]}" -shared -o "$STAGE/lib/arm64-v8a/lib$NAME.so" "${SRCS[@]}" \
   -L"$STUB" -lvm_payload -llog -lm -ldl -Wl,-soname,lib$NAME.so
for x in "${EXTRA_LIBS[@]:-}"; do [ -n "$x" ] && cp "$x" "$STAGE/lib/arm64-v8a/"; done
# stripped copies: the dynamic symbol table (what dlopen/dlsym need) stays, the rest of libllama's 40 MB goes
for x in "$STAGE"/lib/arm64-v8a/*.so; do "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded "$x"; done
for x in "${EXTRA_ASSETS[@]:-}"; do [ -n "$x" ] && cp "$x" "$STAGE/assets/model.calib"; done
echo "payload: $(stat -c %s "$STAGE/lib/arm64-v8a/lib$NAME.so") bytes"
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf" -d "$STAGE/lib/arm64-v8a/lib$NAME.so" | grep -E 'NEEDED' | sed 's/^/  /'

# --- 3. the host app: one activity, system API via reflection -> classes.dex --
JAVAC="${JAVAC:-javac}"
mkdir -p "$STAGE/classes" "$STAGE/dex"
"$JAVAC" --release 17 -Xlint:-options -cp "$SDK/platforms/android-$API/android.jar" -d "$STAGE/classes" "$HERE"/host/app/*.java
"$BT/d8" --min-api 34 --output "$STAGE/dex" "$STAGE"/classes/host/enclave/anchor/avf/*.class 2>&1 | grep -v '^Warning' || true
[ -f "$STAGE/dex/classes.dex" ] || { echo "d8 produced no classes.dex" >&2; exit 2; }
echo "dex: $(stat -c %s "$STAGE/dex/classes.dex") bytes"

# --- 4. the APK: manifest via aapt2, dex + native lib stored uncompressed ----
cd "$STAGE"
"$BT/aapt2" link -o unaligned.apk --manifest "$HERE/AndroidManifest.xml" \
   -I "$SDK/platforms/android-$API/android.jar" --min-sdk-version 34 --target-sdk-version $API
# extractNativeLibs=false demands STORED (-0) entries, page-aligned by zipalign -p
python3 - "$NAME" <<'PYZ'
import sys, zipfile
name = sys.argv[1]
import glob, os
with zipfile.ZipFile("unaligned.apk", "a", compression=zipfile.ZIP_STORED) as z:
    z.write("dex/classes.dex", "classes.dex", compress_type=zipfile.ZIP_STORED)
    for so in sorted(glob.glob("lib/arm64-v8a/*.so")):
        z.write(so, so, compress_type=zipfile.ZIP_STORED)
    for a in sorted(glob.glob("assets/*")):
        z.write(a, a, compress_type=zipfile.ZIP_STORED)
PYZ
"$BT/zipalign" -p -f 4 unaligned.apk aligned.apk
"$BT/apksigner" sign --ks "$HERE/keys/anchor.jks" --ks-pass pass:anchor123 --ks-key-alias anchor \
   --v1-signing-enabled false --v2-signing-enabled true --v3-signing-enabled true --v4-signing-enabled true \
   --out "$OUT/$NAME.apk" aligned.apk
# the v4 signature's Merkle root IS the pVM's codeHash for this apk (pins.py); vm run-app takes the file as its idsig
"$BT/apksigner" verify --print-certs "$OUT/$NAME.apk" | grep -E 'SHA-256|Verified' | sed 's/^/  /'
echo "APK: $OUT/$NAME.apk ($(stat -c %s "$OUT/$NAME.apk") bytes)"
