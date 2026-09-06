#!/usr/bin/env python3
"""The dealer's loop (shielded/dealer/PLAN.md P3): keep one pVM's bank of pad
shipments ahead of its ledger mark, mint what is missing, prune what is spent.

  dealer-loop.py --relay https://api.enclave.host --name pixel-1 --master <hex64> \
                 --model model.gguf --calib model.calib --out /bank [--ahead 256] [--chunk 64] [--once]
  dealer-loop.py --seed <hex64> --seed-id <hex32> --pk <hex64> --mark N ...      (offline: no relay)
  dealer-loop.py --derive-only --master <hex64> --keyfp <hex64> [--epoch 1]     (prints seed + seed_id)
  dealer-loop.py --plan-only --out DIR --seed-id ID --mark N [--ahead --chunk]   (prints the ranges it would mint)

Environment for the minting tool: DEALER (path to shielded-dealer), SHIELDED_SO,
GGML_CPU_SO, LD_LIBRARY_PATH. The seed is derived exactly as relay/pads.mjs does
(HKDF-SHA512, salt = keyFp bytes, info "enclave-pads-seed:<epoch>"); the master
seed is the platform's secret and never leaves the dealer's process.
"""
import argparse, hashlib, hmac, json, os, re, subprocess, sys, time, urllib.request


def hkdf_sha512(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha512).digest()
    out, t, ctr = b"", b"", 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([ctr]), hashlib.sha512).digest()
        out += t; ctr += 1
    return out[:length]


def derive_seed(master_hex: str, keyfp_hex: str, epoch: int = 1):
    master, keyfp = bytes.fromhex(master_hex), bytes.fromhex(keyfp_hex)
    seed = hkdf_sha512(master, keyfp, f"enclave-pads-seed:{epoch}".encode(), 32)
    seed_id = hkdf_sha512(master, keyfp, f"enclave-pads-seed-id:{epoch}".encode(), 16).hex()
    return seed.hex(), seed_id


NAME_RE = re.compile(r"^([0-9a-f]{32})-(\d+)-(\d+)\.pads$")


def shipments(out_dir: str, seed_id: str):
    """(index0, count, path) of every shipment of this seed in the bank."""
    found = []
    for n in os.listdir(out_dir) if os.path.isdir(out_dir) else []:
        m = NAME_RE.match(n)
        if m and m.group(1) == seed_id:
            found.append((int(m.group(2)), int(m.group(3)), os.path.join(out_dir, n)))
    return sorted(found)


def plan(existing, mark: int, ahead: int, chunk: int):
    """Chunk-aligned ranges [i0, i0+chunk) needed to cover [mark, mark+ahead) that no
    shipment covers entirely, and the shipments wholly below the mark (spent)."""
    covered = [(i0, i0 + c) for i0, c, _ in existing]
    want, start = [], mark - mark % chunk
    while start < mark + ahead:
        if not any(lo <= start and start + chunk <= hi for lo, hi in covered):
            want.append((start, chunk))
        start += chunk
    spent = [p for i0, c, p in existing if i0 + c <= mark]
    return want, spent


def relay_get(base: str, path: str):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay"); ap.add_argument("--name")
    ap.add_argument("--master"); ap.add_argument("--keyfp"); ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--seed"); ap.add_argument("--seed-id"); ap.add_argument("--pk"); ap.add_argument("--mark", type=int)
    ap.add_argument("--model"); ap.add_argument("--calib"); ap.add_argument("--out")
    ap.add_argument("--ahead", type=int, default=256); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--once", action="store_true"); ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--derive-only", action="store_true"); ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    if a.derive_only:
        seed, seed_id = derive_seed(a.master, a.keyfp, a.epoch)
        print(json.dumps({"seed": seed, "seed_id": seed_id})); return 0

    while True:
        seed, seed_id, pk, mark = a.seed, a.seed_id, a.pk, a.mark
        if a.relay and a.name:
            info = relay_get(a.relay, f"/v1/pads/pvm?name={a.name}")
            seed_id, pk, mark = info["seed_id"], info.get("padKey") or pk, info.get("mark", 0)
            if a.master:
                seed, sid = derive_seed(a.master, info["keyFp"], info.get("epoch", a.epoch))
                if sid != seed_id: sys.exit(f"seed id mismatch: relay {seed_id}, derived {sid} (wrong master or epoch?)")
        if not seed_id or mark is None:
            sys.exit("need --relay/--name or --seed-id/--mark")
        existing = shipments(a.out, seed_id)
        want, spent = plan(existing, mark, a.ahead, a.chunk)
        if a.plan_only:
            print(json.dumps({"seed_id": seed_id, "mark": mark, "mint": want, "prune": spent})); return 0
        for p in spent:
            os.unlink(p); print(f"pruned {os.path.basename(p)} (below mark {mark})", flush=True)
        if want:
            if not (seed and pk and a.model and a.calib):
                sys.exit("minting needs --seed (or --master) and --pk (or a pVM with a pad key), --model, --calib")
            dealer = os.environ.get("DEALER", "shielded-dealer")
            ranges = ",".join(f"{i0}:{c}" for i0, c in want)
            tmpl = os.path.join(a.out, f"{seed_id}-{{index0}}-{{count}}.pads")
            cmd = [dealer, a.model, "--out", tmpl, "--seed", seed, "--seed-id", seed_id, "--pk", pk, "--ranges", ranges]
            env = {**os.environ, "SHIELDED_CALIB": a.calib}
            t0 = time.time()
            r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            if r.returncode != 0: sys.exit(f"shielded-dealer failed ({r.returncode}) for {ranges}")
            print(f"minted {len(want)} shipment(s) for {seed_id} covering up to {want[-1][0] + want[-1][1]} in {time.time() - t0:.1f} s", flush=True)
        else:
            print(f"bank for {seed_id} covers [{mark}, {mark + a.ahead}); nothing to mint", flush=True)
        if a.once: return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
