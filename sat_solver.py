#!/usr/bin/env python3
"""Claude's numba local-search solver (reconstructed from flattened paste).
python3 sat_solver.py clauses.json output.txt 570 [n_vars] [n_proc]
exit 0 found (verified, written), 1 not found, 2 bad input."""
import sys, json, time, os, multiprocessing as mp
import numpy as np
import numba as nb

_DEADLINE = None
def load_clauses(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "gz" in data:
        import gzip, base64
        global _DEADLINE
        _DEADLINE = data.get("deadline_unix")
        data = json.loads(gzip.decompress(base64.b64decode(data["gz"])))
    if isinstance(data, dict):
        for k in ("clauses", "cnf", "formula"):
            if k in data:
                data = data[k]; break
    return [[int(x) for x in c] for c in data]

def prepare(n, clauses):
    cl = []
    for c in clauses:
        s = []; seen = set(); taut = False
        for l in c:
            if -l in seen:
                taut = True; break
            if l not in seen:
                seen.add(l); s.append(l)
        if not taut:
            cl.append(s)
    lens = np.array([len(c) for c in cl], dtype=np.int64)
    off = np.zeros(len(cl) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    flat = np.array([l for c in cl for l in c], dtype=np.int32)
    li = np.where(flat > 0, 2 * (flat - 1), 2 * (-flat - 1) + 1).astype(np.int64)
    cid = np.repeat(np.arange(len(cl), dtype=np.int32), lens)
    order = np.argsort(li, kind="stable")
    occ = cid[order].astype(np.int32)
    occ_start = np.zeros(2 * n + 1, dtype=np.int64)
    np.cumsum(np.bincount(li, minlength=2 * n), out=occ_start[1:])
    maxlen = int(lens.max()) if len(lens) else 1
    return flat, off, occ_start, occ, maxlen

def verify(clauses, bits):
    for c in clauses:
        if not any((l > 0) == (bits[abs(l) - 1] == 1) for l in c):
            return False
    return True

@nb.njit(cache=True)
def seed_rng(s):
    np.random.seed(s)

@nb.njit(cache=True)
def init_state(flat, off, assign, ntrue, unsat, pos):
    m = off.shape[0] - 1
    nu = 0
    for c in range(m):
        t = 0
        for k in range(off[c], off[c + 1]):
            l = flat[k]
            if (l > 0) == (assign[abs(l) - 1] == 1):
                t += 1
        ntrue[c] = t
        if t == 0:
            unsat[nu] = c; pos[c] = nu; nu += 1
        else:
            pos[c] = -1
    return nu

@nb.njit(cache=True)
def make_true(l, occ_start, occ, assign, ntrue, unsat, pos, nu):
    if l > 0:
        li = 2 * (l - 1); assign[l - 1] = 1
    else:
        li = 2 * (-l - 1) + 1; assign[-l - 1] = 0
    ni = li ^ 1
    for k in range(occ_start[li], occ_start[li + 1]):
        cc = occ[k]
        ntrue[cc] += 1
        if ntrue[cc] == 1:
            p = pos[cc]; last = unsat[nu - 1]
            unsat[p] = last; pos[last] = p; pos[cc] = -1
            nu -= 1
    for k in range(occ_start[ni], occ_start[ni + 1]):
        cc = occ[k]
        ntrue[cc] -= 1
        if ntrue[cc] == 0:
            unsat[nu] = cc; pos[cc] = nu; nu += 1
    return nu

@nb.njit(cache=True)
def probsat_chunk(flat, off, occ_start, occ, assign, ntrue, unsat, pos, st, nflips, probs, maxlen):
    nu = st[0]; best = st[1]
    w = np.empty(maxlen)
    done = 0
    for it in range(nflips):
        if nu == 0:
            break
        done += 1
        c = unsat[min(int(np.random.random() * nu), nu - 1)]
        a = off[c]; b = off[c + 1]
        tot = 0.0
        for j in range(b - a):
            l = flat[a + j]
            if l > 0:
                ni = (2 * (l - 1)) ^ 1
            else:
                ni = (2 * (-l - 1) + 1) ^ 1
            br = 0
            for k in range(occ_start[ni], occ_start[ni + 1]):
                if ntrue[occ[k]] == 1:
                    br += 1
            if br > 255:
                br = 255
            w[j] = probs[br]; tot += w[j]
        r = np.random.random() * tot
        j = 0; acc = w[0]
        while r > acc and j < b - a - 1:
            j += 1; acc += w[j]
        nu = make_true(flat[a + j], occ_start, occ, assign, ntrue, unsat, pos, nu)
        if nu < best:
            best = nu
    st[0] = nu; st[1] = best
    return done

@nb.njit(cache=True)
def wls_chunk(flat, off, occ_start, occ, assign, ntrue, unsat, pos, wt, st, nflips, pflat, pnoise, smooth_every):
    nu = st[0]; best = st[1]; bumps = st[2]
    m = off.shape[0] - 1
    done = 0
    for it in range(nflips):
        if nu == 0:
            break
        done += 1
        c = unsat[min(int(np.random.random() * nu), nu - 1)]
        a = off[c]; b = off[c + 1]
        bj = 0; bs = -2000000000; cnt = 0
        for j in range(b - a):
            l = flat[a + j]
            if l > 0:
                li = 2 * (l - 1)
            else:
                li = 2 * (-l - 1) + 1
            ni = li ^ 1
            mk = 0
            for k in range(occ_start[li], occ_start[li + 1]):
                cc = occ[k]
                if ntrue[cc] == 0:
                    mk += wt[cc]
            bk = 0
            for k in range(occ_start[ni], occ_start[ni + 1]):
                cc = occ[k]
                if ntrue[cc] == 1:
                    bk += wt[cc]
            s = mk - bk
            if s > bs:
                bs = s; bj = j; cnt = 1
            elif s == bs:
                cnt += 1
                if int(np.random.random() * cnt) == 0:
                    bj = j
        r = np.random.random()
        if bs > 0 or (bs == 0 and r < pflat):
            nu = make_true(flat[a + bj], occ_start, occ, assign, ntrue, unsat, pos, nu)
        elif r < pnoise:
            j = min(int(np.random.random() * (b - a)), b - a - 1)
            nu = make_true(flat[a + j], occ_start, occ, assign, ntrue, unsat, pos, nu)
        else:
            for i in range(nu):
                wt[unsat[i]] += 1
            bumps += 1
            if bumps % smooth_every == 0:
                for cc in range(m):
                    if wt[cc] > 1:
                        wt[cc] -= 1
        if nu < best:
            best = nu
    st[0] = nu; st[1] = best; st[2] = bumps
    return done

def probs_table(cb, eps):
    return (eps + np.arange(256, dtype=np.float64)) ** (-cb)

def run_worker(idx, n, P, deadline, q, stop, seed, verbose):
    flat, off, occ_start, occ, maxlen = P
    m = off.shape[0] - 1
    mode = idx % 3
    seed_rng(seed)
    rng = np.random.default_rng(seed)
    cb, eps = 2.06, 0.9
    if mode == 0:
        restart_every = 0
    elif mode == 1:
        restart_every = 150_000_000
    else:
        restart_every, cb = 400_000_000, 2.3
    probs = probs_table(cb, eps)
    chunk = 2_000_000 if mode != 1 else 1_000_000
    t0 = time.time(); total = 0; last_log = t0; run = 0
    while time.time() < deadline and not stop.is_set():
        run += 1
        assign = rng.integers(0, 2, n).astype(np.int8)
        ntrue = np.zeros(m, np.int32); unsat = np.zeros(m, np.int32)
        pos = np.full(m, -1, np.int32); wt = np.ones(m, np.int32)
        st = np.zeros(3, np.int64)
        st[0] = init_state(flat, off, assign, ntrue, unsat, pos); st[1] = st[0]
        flips_run = 0
        while time.time() < deadline and not stop.is_set():
            if mode == 1:
                d = wls_chunk(flat, off, occ_start, occ, assign, ntrue, unsat, pos, wt, st, chunk, 0.15, 0.02, 50)
            else:
                d = probsat_chunk(flat, off, occ_start, occ, assign, ntrue, unsat, pos, st, chunk, probs, maxlen)
            total += d; flips_run += d
            if st[0] == 0:
                q.put((idx, assign.tobytes())); return
            now = time.time()
            if verbose and now - last_log > 20:
                last_log = now
                print("[w%d mode%d run%d] %.0fs %.2fM flips/s now=%d best=%d" % (idx, mode, run, now - t0, total / (now - t0) / 1e6, st[0], st[1]), file=sys.stderr, flush=True)
            if restart_every and flips_run >= restart_every:
                break

def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(2)
    path_json, path_out = sys.argv[1], sys.argv[2]
    limit = float(sys.argv[3])
    t_start = time.time(); deadline = t_start + limit
    clauses = load_clauses(path_json)
    if _DEADLINE:
        deadline = min(deadline, float(_DEADLINE))
        print("deadline in %.0fs" % (deadline - time.time()), file=sys.stderr, flush=True)
    n = max((abs(l) for c in clauses for l in c), default=0)
    if len(sys.argv) > 4 and sys.argv[4]:
        n = max(n, int(sys.argv[4]))
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    if any(len(c) == 0 for c in clauses):
        sys.exit(2)
    P = prepare(n, clauses)
    tiny = [[1, 2, 3], [-1, 2], [-2, 3], [1, -3]]
    Pt = prepare(3, tiny)
    for fn in ("p", "w"):
        a = np.zeros(3, np.int8)
        nt = np.zeros(4, np.int32); un = np.zeros(4, np.int32)
        ps = np.full(4, -1, np.int32); wtt = np.ones(4, np.int32)
        stt = np.zeros(3, np.int64)
        stt[0] = init_state(Pt[0], Pt[1], a, nt, un, ps); stt[1] = stt[0]
        if fn == "p":
            probsat_chunk(Pt[0], Pt[1], Pt[2], Pt[3], a, nt, un, ps, stt, 50, probs_table(2.06, 0.9), Pt[4])
        else:
            wls_chunk(Pt[0], Pt[1], Pt[2], Pt[3], a, nt, un, ps, wtt, stt, 50, 0.15, 0.02, 50)
    print("compiled %.1fs n=%d m=%d" % (time.time() - t_start, n, len(clauses)), file=sys.stderr, flush=True)
    ctx = mp.get_context("fork")
    q = ctx.Queue(); stop = ctx.Event()
    base = int.from_bytes(os.urandom(3), 'little')
    procs = []
    for i in range(nproc):
        p = ctx.Process(target=run_worker, args=(i, n, P, deadline, q, stop, base + 7919 * i, True))
        p.daemon = True; p.start(); procs.append(p)
    result = None
    while time.time() < deadline + 2:
        try:
            idx, raw = q.get(timeout=0.5)
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
            continue
        bits = np.frombuffer(raw, dtype=np.int8)
        if verify(clauses, bits):
            result = bits
            print("worker %d solved at %.1fs (verified)" % (idx, time.time() - t_start), file=sys.stderr)
            break
        print("worker %d reported solution but verify FAILED" % idx, file=sys.stderr)
    stop.set()
    for p in procs:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()
    if result is None:
        print("not found", file=sys.stderr); sys.exit(1)
    s = "".join("1" if x else "0" for x in result[:n])
    tmp = path_out + ".tmp"
    with open(tmp, "w") as f:
        f.write(s)
    os.replace(tmp, path_out)
    print("written %s (%d chars)" % (path_out, len(s)), file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    main()
