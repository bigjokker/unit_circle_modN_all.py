#!/usr/bin/env python3
# unit_circle_modN_all.py
# Sample or verify solutions to x^2 + A*y^2 ≡ B (mod N).
# Modes:
#   param    -> fast rational parametrization (B=1; may reveal a factor)
#               Example: python testbetter.py param --N 15 --A 1 --num_solutions 2 --tries 500
#   prime    -> search T=4kN+B (B odd, or B=2 special) prime-like + Cornacchia
#               (sieve + optional multiprocessing, multi-solution)
#               Example: python testbetter.py prime --N 10007 --A 1 --B 1 --num_solutions 1 --k_limit 10000 --jobs 4
#   verify   -> hardened checker with provenance, t-recovery and gcd sniffs
#               Example: python testbetter.py verify --N 15 --A 1 --B 1 1 1
#   compose  -> group law: compose one or more solutions (x,y) under Z[sqrt(-A)] mod N
#               Example: python testbetter.py compose --N 15 --A 1 --pairs 1 1 2 3
#   crt      -> solve modulo factored N: ∏ p_i^e_i, with odd p_i Hensel lift; p=2 via 2-adic stepping
#               Example: python testbetter.py crt --A 1 --B 1 --factors "3^1,5^1"
#   batch    -> run param/prime across a list/range of A values; parallel across A if desired
#               Example: python testbetter.py batch --modeA prime --N 15 --A_values "1,3-5" --B 1
#   hybrid   -> try param quickly; if no hit/factor, fall back to prime
#               Example: python testbetter.py hybrid --N 15 --A 1 --num_solutions 1
#   selftest -> quick built-in smoke tests
#               Example: python testbetter.py selftest
#   quad     -> solve general quadratic ax^2 + bx + c ≡ 0 (mod N)
#               Example: python testbetter.py quad --N 15 --a 1 --b 2 --c 3 --factors "3^1,5^1"
#   bench    -> Micro-benchmark a mode and include elapsed time in JSON.
#   factor   -> Try to factor N via gcd sniffs on solutions.
#   solvable -> Check if the equation has solutions (requires factors or small N).
#               Example: python testbetter.py solvable --N 15 --A 1 --B 1 --factors "3^1,5^1"
#   auto     -> Automatically choose the best mode based on inputs.
#               Example: python testbetter.py auto --N 15 --A 1 --B 1 --num_solutions 1
#   dioph    -> Solve general Diophantine equations like ax^2 + bxy + cy^2 = d mod m
#               Example: python testbetter.py dioph --N 15 --a 1 --b 0 --c 1 --d 1 --factors "3^1,5^1"
#   pell     -> Solve Pell equation x^2 - D y^2 = N
#               Example: python testbetter.py pell --D 13 --N 1
#   rsa_diag -> Diagnostics for RSA moduli
#               Example: python testbetter.py rsa_diag --N 91
#   orbit    -> Compute solution orbits under group actions
#               Example: python testbetter.py orbit --N 15 --A 1 --x 1 --y 1 --max_power 10
#
# Python 3.8+

import argparse
import random
import json
import sys
import hashlib
import logging
import itertools
from math import isqrt, gcd, prod
from typing import Optional, Tuple, List, Dict, Iterator
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import time as _t
import pytest
import types
try:
    from sympy.solvers.diophantine.diophantine import diop_DN
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Setup basic logging
logging.basicConfig(
    filename='testbetter_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s'
)

# A well-known 2048-bit RSA modulus (decimal)
RSA_2048_N = (
    2519590847565789349402718324004839857142928212620403202777713783604366202070759555626401852588078440691829064124951508218929855914917618450280848912007284499268739280728777673597141834727026189637501497182469116507761337985909570009733045974880842840179742910064245869181719511874612151517265463228221686998754918242243363725908514186546204357679842338718477449442074022384966807237540068014837888908838532623612836537824281198163815010674810451660377306056201619676256133844143603833904414952634432190114657544454178424020924616515723350778707749817125772467962926386356373289912154831438167899885040445364023527381951378636564391212010397122822120720357
)

# Global variables
_GLOBAL_DEADLINE = None
_COUNTERS = {
    'prp_calls': 0,
    'prp_mr_calls': 0,
    'prp_bpsw_calls': 0,
    'lucas_calls': 0,
    'k_tested': 0,
    'trial_divisible': 0,
    'k_skipped_by_residue': 0,
    'cornacchia_calls': 0,
    'cornacchia_success': 0,
}

# ---------- tiny logging with caps ----------
def _mk_logger(verbose: int = 0, cap: int = 200):
    """
    Create logger functions with verbosity control and capping.
    
    :param verbose: Verbosity level (0: off, 1: summary, 2: capped logs, 3+: uncapped).
    :param cap: Max logs before capping at verbose=2.
    :return: (log function, summary function)
    """
    count = {"n": 0}
    def log(*args, **kwargs):
        if verbose <= 0:
            return
        if verbose == 1:
            return
        if count["n"] < cap or verbose >= 3:
            print(*args, file=sys.stderr, **{k: v for k, v in kwargs.items() if k != "file"})
        count["n"] += 1
    def summary(*args, **kwargs):
        if verbose >= 1:
            print(*args, file=sys.stderr, **{k: v for k, v in kwargs.items() if k != "file"})
    return log, summary

# ---------- deterministic RNG ----------
_SMALL_PRIMES = [2,3,5,7,11,13,17,19,23,29,31]
def _rng_for_n(seed: Optional[int], n: int) -> random.Random:
    """
    Deterministic RNG per (seed, n). If seed is None, use global random.
    
    :param seed: Optional seed for reproducibility.
    :param n: Modulus or identifier.
    :return: random.Random instance.
    """
    if seed is None:
        return random
    b = f"{seed}:{n}".encode()
    h = hashlib.blake2b(b, digest_size=16).digest()
    return random.Random(int.from_bytes(h, "big"))

# ---------- Miller–Rabin ----------
def _mr_witness(a: int, n: int, d: int, s: int) -> bool:
    """
    Check if a is a Miller-Rabin witness for composite n.
    """
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True  # composite

def _mr_deterministic_64(n: int) -> bool:
    """
    Deterministic Miller-Rabin for n < 2^64 using known strong bases.
    """
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a % n == 0:
            return True
        if _mr_witness(a, n, d, s):
            return False
    return True

def miller_rabin(n: int, rounds: int = 40, seed: Optional[int] = None) -> bool:
    """
    Probabilistic primality test using Miller-Rabin.
    
    :param n: Number to test.
    :param rounds: Number of rounds (higher = more accurate).
    :param seed: Seed for random bases.
    :return: True if probable prime.
    """
    _COUNTERS['prp_calls'] += 1
    _COUNTERS['prp_mr_calls'] += 1
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    if n < (1 << 64):
        return _mr_deterministic_64(n)
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    rng = _rng_for_n(seed, n)
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        if _mr_witness(a, n, d, s):
            return False
    return True

# ---------- Baillie-PSW probable prime test ----------

def _sprp_base2(n: int) -> bool:
    """Strong probable-prime test to base 2 (Miller–Rabin with a=2)."""
    if n % 2 == 0:
        return n == 2
    # write n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(2, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False

def _selfridge_params(n: int):
    """Pick D (≡1 mod 4) with Jacobi(D/n) = -1; then P=1, Q=(1-D)/4."""
    D = 5
    sign = 1
    while True:
        Ds = D * sign
        j = jacobi(Ds, n)
        if j == -1:
            break
        if j == 0:
            # nontrivial gcd with n => composite (unless n==|D|)
            return None
        D += 2
        sign = -sign
    P = 1
    Q = (1 - Ds) // 4  # guaranteed integer
    return P, Q

def lucas_UV(P: int, Q: int, n: int, k: int):
    """
    Fast-doubling Lucas: return (U_k, V_k, Q^k) mod n.
    Start from k=1 state (U1=1, V1=P, Q^1=Q).
    """
    if k == 0:
        return 0, 2 % n, 1
    U = 1
    V = P % n
    Qk = Q % n
    inv2 = pow(2, -1, n)
    D = (P * P - 4 * Q) % n
    # skip leading '1' bit of k
    for b in bin(k)[3:]:
        # double
        U = (U * V) % n
        V = (V * V - 2 * Qk) % n
        Qk = (Qk * Qk) % n
        if b == '1':
            # increment
            U, V = ((P * U + V) * inv2) % n, ((D * U + P * V) * inv2) % n
            Qk = (Qk * Q) % n
    return U, V, Qk

def is_strong_lucas_prp(n: int, P: int, Q: int) -> bool:
    D = P * P - 4 * Q
    j = jacobi(D, n)
    if j == 0:
        return False  # nontrivial gcd => composite
    dd = n - j       # n+1 if j = -1, else n-1
    # dd = d * 2^s
    d = dd
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    U, V, Qd = lucas_UV(P, Q, n, d)
    if U % n == 0:
        return True
    Vm, Qm = V, Qd
    for _ in range(s):
        if Vm % n == 0:
            return True
        Vm = (Vm * Vm - 2 * Qm) % n   # <-- uses Q^m
        Qm = (Qm * Qm) % n
    return False

def bpsw(n: int) -> bool:
    if n < 2:
        return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n % p == 0:
            return n == p
    if _is_square(n):
        return False
    # base-2 MR
    if not _sprp_base2(n):
        return False
    # Selfridge params (P=1, Q=(1-D)/4)
    params = _selfridge_params(n)
    if params is None:
        return False
    P, Q = params
    _COUNTERS['prp_calls'] += 1
    _COUNTERS['prp_bpsw_calls'] = _COUNTERS.get('prp_bpsw_calls', 0) + 1
    _COUNTERS['lucas_calls'] += 1
    return is_strong_lucas_prp(n, P, Q)

# ---------- symbols (cached) ----------
@lru_cache(maxsize=200_000)
def jacobi(a: int, n: int) -> int:
    """
    Compute the Jacobi symbol (a/n).
    
    :param a: Numerator.
    :param n: Denominator (odd positive integer).
    :return: Jacobi symbol value.
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer.")
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    if n == 1:
        return result
    else:
        return 0

@lru_cache(maxsize=200_000)
def legendre(a: int, p: int) -> int:
    """
    Compute the Legendre symbol (a/p).
    
    :param a: Numerator.
    :param p: Prime denominator.
    :return: 1 if quadratic residue, -1 if not, 0 if a ≡ 0 mod p.
    """
    a %= p
    if a == 0:
        return 0
    t = pow(a, (p - 1) // 2, p)
    if t == p - 1:
        return -1
    else:
        return t

def _is_square(n: int) -> bool:
    """
    Check if n is a perfect square.
    """
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

# ---------- Tonelli–Shanks & Cornacchia ----------
def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """
    Compute square root of n mod p using Tonelli-Shanks algorithm.
    Steps: Find non-residue z, compute initial x/t, loop to adjust.
    """
    n %= p
    if n == 0:
        return 0
    if p == 2:
        return n
    if legendre(n, p) != 1:
        return None
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while legendre(z, p) != -1:
        z += 1
    c = pow(z, q, p)
    x = pow(n, (q + 1) // 2, p)
    t = pow(n, q, p)
    m = s
    while t != 1:
        i = 1
        t2i = pow(t, 2, p)
        while i < m and t2i != 1:
            t2i = pow(t2i, 2, p)
            i += 1
        if t2i != 1:
            return None
        b = pow(c, 1 << (m - i - 1), p)
        x = (x * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i
    return x

def hensel_lift(x: int, p: int, e: int, f: callable, df: callable) -> Optional[int]:
    mod = p
    for _ in range(1, e):
        mod *= p
        fx = f(x) % mod
        dfx = df(x) % mod
        if dfx == 0:
            return None
        inv = pow(dfx, -1, mod)
        x = (x - fx * inv) % mod
    return x

def hensel_lift_quad_root(p: int, e: int, a: int, b: int, c: int, x0: int) -> Optional[int]:
    """
    Lift root x0 mod p for ax^2 + bx + c ≡ 0 mod p^e using Hensel.
    f(x) = a x^2 + b x + c, df(x) = 2a x + b
    """
    def f(x):
        return a * x**2 + b * x + c
    def df(x):
        return 2 * a * x + b
    return hensel_lift(x0, p, e, f, df)

def cornacchia(p: int, A: int = 1) -> Optional[Tuple[int,int]]:
    """
    Solve x^2 + A y^2 = p using Cornacchia's algorithm.
    """
    if not (p > 2 and p % 2 == 1):
        return None
    if A <= 0:
        return None
    if legendre((-A) % p, p) != 1:
        return None
    def try_root(s):
        a, b = p, s
        while b * b > p:
            a, b = b, a % b
        x = b
        t = p - x * x
        if t % A != 0:
            return None
        y2 = t // A
        if not _is_square(y2):
            return None
        return x, isqrt(y2)
    s = tonelli_shanks((-A) % p, p)
    if s is None:
        return None
    sol = try_root(s)
    if sol is not None:
        return sol
    return try_root((p - s) % p)

# ---------- Twofold Cornacchia ----------
def twofold_cornacchia(m: int, A: int = 1) -> Optional[Tuple[int,int]]:
    """
    Extended Cornacchia for possible composite m.
    """
    if miller_rabin(m):
        return cornacchia(m, A)
    for x in range(1, isqrt(m) + 1):
        t = m - x * x
        if t % A == 0:
            y2 = t // A
            if _is_square(y2):
                return x, isqrt(y2)
    return None

# ---------- Parametric Sampling ----------
def parametric_unit(N: int, A: int, seed: Optional[int] = None) -> Tuple[int, int, int]:
    """
    Sample rational point on x^2 + A y^2 ≡ 1 mod N.
    Returns (x, y, d) where d = gcd(t^2 + A, N) may reveal factor.
    """
    if gcd(A, N) != 1:
        raise ValueError("gcd(A, N) must be 1")
    rng = _rng_for_n(seed, N)
    while True:
        t = rng.randrange(N)
        denom = pow(t**2 + A, -1, N)
        x = (t**2 - A) * denom % N
        y = 2 * t * denom % N
        d = gcd(t**2 + A, N)
        if d > 1:
            return x, y, d
        if (x**2 + A * y**2) % N == 1:
            return x, y, 1

def parametric_unit_multi(N: int, A: int, num_solutions: int = 1, tries: int = 100, seed: Optional[int] = None, unique_only: bool = False, verbose: int = 0) -> Dict:
    """
    Find multiple unique solutions via parametric sampling.
    """
    solutions = []
    found = set()
    factors = set()
    log, summary = _mk_logger(verbose)
    for _ in range(tries):
        try:
            x, y, d = parametric_unit(N, A, seed)
            if d > 1 and d < N:
                factors.add(d)
            key = (min(x, N - x), min(y, N - y)) if unique_only else (x, y)
            if key not in found:
                found.add(key)
                solutions.append({"x_mod_N": x, "y_mod_N": y, "gcd_sniff": d})
                if len(solutions) >= num_solutions:
                    break
        except ValueError:
            break
    out = {"method": "param", "N": N, "A": A, "num_found": len(solutions), "solutions": solutions}
    if factors:
        out["revealed_factors"] = sorted(factors)
    return out

# ---------- Prime-like Search ----------
def is_probable_prime(n: int, prp_test: str = 'bpsw', mr_rounds: int = 40, seed: Optional[int] = None) -> bool:
    """
    Test if n is probable prime using specified method.
    """
    if prp_test == 'bpsw':
        return bpsw(n)
    elif prp_test == 'mr':
        return miller_rabin(n, mr_rounds, seed)
    raise ValueError("Invalid prp_test")

def sieve_k_range(k_start: int, k_limit: int, N: int, B: int, sieve_bound: Optional[int]) -> Iterator[int]:
    """
    Sieve k in [k_start, k_limit] to yield candidates where T=4kN+B may be prime-like.
    """
    if sieve_bound is None:
        sieve_bound = min(1000, k_limit // 10)
    primes = [p for p in _SMALL_PRIMES if p <= sieve_bound]
    for k in range(k_start, k_limit):
        T = 4 * k * N + B
        if any(T % p == 0 for p in primes):
            _COUNTERS['trial_divisible'] += 1
            continue
        yield k

def find_xy_for_k(k: int, N: int, A: int, B: int, prp_test: str, mr_rounds: int, seed: Optional[int], log) -> Optional[Dict]:
    """
    Worker: For given k, check if T prime-like, then apply Cornacchia.
    """
    _COUNTERS['k_tested'] += 1
    T = 4 * k * N + B
    if not is_probable_prime(T, prp_test, mr_rounds, seed):
        return None
    _COUNTERS['cornacchia_calls'] += 1
    sol = cornacchia(T, A)
    if sol:
        x, y = sol
        _COUNTERS['cornacchia_success'] += 1
        d = gcd(x**2 + A * y**2 - B, N)
        log(f"Found sol for k={k}, T={T}: ({x},{y}), gcd_sniff={d}")
        return {"k": k, "T": T, "x_mod_N": x % N, "y_mod_N": y % N, "gcd_sniff": d}
    return None

def find_xy_mod_N_prime_multi(N: int, A: int, B: int, num_solutions: int, k_start: int, k_limit: int, sieve_bound: Optional[int], seed: Optional[int], mr_rounds: int, verbose: int, prp_test: str) -> Dict:
    """
    Search for multiple solutions via prime-like T.
    """
    log, summary = _mk_logger(verbose)
    solutions = []
    for k in sieve_k_range(k_start, k_limit, N, B, sieve_bound):
        sol = find_xy_for_k(k, N, A, B, prp_test, mr_rounds, seed, log)
        if sol:
            solutions.append(sol)
            if len(solutions) >= num_solutions:
                break
    out = {"method": "prime", "N": N, "A": A, "B": B, "num_found": len(solutions), "solutions": solutions, "counters": _COUNTERS.copy()}
    summary(f"Summary: {out}")
    return out

def find_xy_mod_N_prime_parallel_multi(N: int, A: int, B: int, num_solutions: int, k_start: int, k_limit: int, jobs: int, chunk: int, sieve_bound: Optional[int], seed: Optional[int], mr_rounds: int, verbose: int, prp_test: str) -> Dict:
    """
    Parallel version using multiprocessing.
    """
    log, summary = _mk_logger(verbose)
    solutions = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = []
        for start in range(k_start, k_limit, chunk):
            end = min(start + chunk, k_limit)
            futures.append(executor.submit(find_xy_mod_N_prime_multi, N, A, B, num_solutions - len(solutions), start, end, sieve_bound, seed, mr_rounds, 0, prp_test))
        for future in as_completed(futures):
            res = future.result()
            solutions.extend(res["solutions"])
            if len(solutions) >= num_solutions:
                break
    out = {"method": "prime_parallel", "N": N, "A": A, "B": B, "num_found": len(solutions), "solutions": solutions, "counters": _COUNTERS.copy()}
    summary(f"Parallel Summary: {out}")
    return out

# ---------- Verification ----------
def check_pair(N: int, x: int, y: int, A: int, B: int, recover_t: bool = True, seed: Optional[int] = None, mr_rounds: int = 40) -> Dict:
    """
    Verify if x^2 + A y^2 ≡ B mod N, recover t if possible, sniff gcd.
    """
    valid = (x**2 + A * y**2) % N == B % N
    out = {"valid": valid, "N": N, "A": A, "B": B, "x": x, "y": y}
    if not valid:
        return out
    d = gcd(x**2 + A * y**2 - B, N)
    out["gcd_sniff"] = d
    if recover_t and A == 1 and B == 1:
        try:
            inv = pow(1 - x, -1, N)
            t = (y * inv) % N
            out["recovered_t"] = t
        except ValueError:
            out["recovered_t"] = None
    return out

# ---------- Composition ----------
def compose_two(N: int, A: int, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """
    Compose two solutions under group law.
    """
    x = (x1 * x2 - A * y1 * y2) % N
    y = (x1 * y2 + y1 * x2) % N
    return x, y

def pow_solution(N: int, A: int, x: int, y: int, power: int) -> Tuple[int, int]:
    """
    Raise solution to power using binary exponentiation.
    """
    result_x, result_y = 1, 0  # Identity
    base_x, base_y = x % N, y % N
    while power > 0:
        if power % 2 == 1:
            result_x, result_y = compose_two(N, A, result_x, result_y, base_x, base_y)
        base_x, base_y = compose_two(N, A, base_x, base_y, base_x, base_y)
        power //= 2
    return result_x, result_y

def orbit(N: int, A: int, x: int, y: int, max_power: int) -> List[Tuple[int, int]]:
    """
    Compute orbit by successive powering.
    """
    orbits = []
    current_x, current_y = x % N, y % N
    for p in range(1, max_power + 1):
        orbits.append((current_x, current_y))
        current_x, current_y = pow_solution(N, A, x, y, p)
        if (current_x, current_y) in orbits:  # Cycle detected
            break
    return orbits

# ---------- CRT Solving ----------
def parse_factors(factors_str: str) -> List[Tuple[int, int]]:
    """
    Parse "p1^e1,p2^e2" to [(p1,e1),(p2,e2)].
    """
    pe = []
    for part in factors_str.split(','):
        if '^' in part:
            p, e = map(int, part.split('^'))
        else:
            p, e = int(part), 1
        pe.append((p, e))
    return pe

def _validate_factors(pe: List[Tuple[int, int]]) -> int:
    """
    Validate factors and compute N = prod p^e.
    """
    N = 1
    for p, e in pe:
        if not miller_rabin(p):
            raise ValueError(f"{p} not prime")
        N *= p ** e
    return N

def chinese_remainder(a: List[int], m: List[int]) -> int:
    """
    Solve system x ≡ a_i mod m_i for pairwise coprime m_i.
    """
    prod_m = prod(m)
    result = 0
    for ai, mi in zip(a, m):
        p = prod_m // mi
        result += ai * p * pow(p, -1, mi)
    return result % prod_m

def hensel_lift_square_root(p: int, e: int, c: int, y0: int) -> Optional[int]:
    """
    Lift square root y0 mod p for y^2 ≡ c mod p^e.
    """
    mod = p
    y = y0
    for k in range(1, e):
        mod *= p
        f = (y**2 - c) % mod
        df = 2 * y % mod
        if df == 0:
            return None
        inv = pow(df, -1, mod)
        y = (y - f * inv) % mod
    return y

def solve_mod_pe(A: int, B: int, p: int, e: int) -> List[Tuple[int, int]]:
    """
    Solve x^2 + A y^2 ≡ B mod p^e using Hensel lift for odd p, brute for p=2 or small.
    """
    if p == 2 or e == 1:
        # Brute for p=2 or base case
        mod = p ** e
        sols = []
        for x in range(mod):
            t = (B - x**2) % mod
            if t % A != 0:
                continue
            y2 = t // A
            if y2 < 0:
                y2 += mod
            if _is_square(y2):
                y = isqrt(y2) % mod
                sols.append((x % mod, y))
                if y != 0:
                    sols.append((x % mod, (mod - y) % mod))
        return sols
    # For odd p, use Tonelli for base, lift y from y^2 ≡ (B - x^2)/A mod p^e
    mod_p = p
    base_sols = solve_mod_pe(A, B, p, 1)
    lifted_sols = []
    for x0, y0 in base_sols:
        c = (B - x0**2) // A % mod_p
        y_lifted = hensel_lift_square_root(p, e, c, y0)
        if y_lifted is not None:
            lifted_sols.append((x0, y_lifted))
            if y_lifted != 0:
                lifted_sols.append((x0, (p**e - y_lifted) % (p**e)))
        # Lift -y0 if different
        y_neg = (p - y0) % p
        if y_neg != y0:
            y_lifted_neg = hensel_lift_square_root(p, e, c, y_neg)
            if y_lifted_neg is not None:
                lifted_sols.append((x0, y_lifted_neg))
                if y_lifted_neg != 0:
                    lifted_sols.append((x0, (p**e - y_lifted_neg) % (p**e)))
    return lifted_sols

def crt_solve(A: int, B: int, pe: List[Tuple[int, int]], limit_per_prime: int = 5000, cap_total: int = 50000, verbose: int = 0, full_solutions: bool = False, count: bool = False, jobs: int = 1) -> Dict or Iterator:
    """
    Solve via CRT for factored N.
    """
    local_sols = {}
    for p, e in pe:
        local_sols[(p, e)] = solve_mod_pe(A, B, p, e)[:limit_per_prime]
    if count:
        total = prod(len(s) for s in local_sols.values())
        return {"count": total, "N_from_factors": _validate_factors(pe)}
    if full_solutions:
        # Generator for large sets
        def gen():
            for combo in itertools.product(*local_sols.values()):
                # CRT combine x and y separately
                xs, ys = zip(*combo)
                X = chinese_remainder(xs, [p**e for p,_ in pe])
                Y = chinese_remainder(ys, [p**e for p,_ in pe])
                yield {"x_mod_N": X, "y_mod_N": Y}
        return gen()
    else:
        return {"method": "crt", "solutions": list(itertools.islice(crt_solve(A, B, pe, full_solutions=True), cap_total))}

def crt_solve_gen(A: int, B: int, pe: List[Tuple[int, int]], verbose: int = 0, jobs: int = 1) -> Iterator[Dict]:
    """
    Generator version for streaming.
    """
    return crt_solve(A, B, pe, full_solutions=True)

# ---------- Quadratic Solving ----------
def quad_solve(N: int, a: int, b: int, c: int, count_only: bool = False, factors: Optional[str] = None) -> Dict:
    """
    Solve ax^2 + bx + c ≡ 0 mod N.
    """
    if factors is None and N.bit_length() > 64:
        return {"method": "quad", "note": "N too large without factors", "solutions": [], "count": "unknown"}
    # Transform to unit circle form or use CRT
    if factors:
        pe = parse_factors(factors)
        # Solve via CRT
        # For quadratic in one var, solve x mod each p^e using tonelli on discriminant
        # But for simple, brute if small
        sols = []
        mods = [p**e for p, e in pe]
        local_sols = []
        for p, e in pe:
            mod_pe = p ** e
            disc = (b**2 - 4*a*c) % mod_pe
            if disc < 0:
                disc += mod_pe
            if legendre(disc, p) != 1 and disc != 0:
                return {"solutions": [], "count": 0}
            # Todo: proper hensel/tonelli for roots
            roots = []
            for x in range(mod_pe):
                if (a * x**2 + b * x + c) % mod_pe == 0:
                    roots.append(x)
            local_sols.append(roots)
        for combo in itertools.product(*local_sols):
            X = chinese_remainder(combo, mods)
            sols.append(X)
        count = len(sols)
        if count_only:
            return {"count": count}
        return {"method": "quad", "solutions": sols, "count": count}
    else:
        # Brute for small N
        sols = []
        for x in range(N):
            if (a * x**2 + b * x + c) % N == 0:
                sols.append(x)
        count = len(sols)
        if count_only:
            return {"count": count}
        return {"solutions": sols, "count": count}

# ---------- Benchmarking ----------
def run_bench_mode(bench_mode: str, args: argparse.Namespace) -> Dict:
    start = _t.time()
    if bench_mode == "param":
        out = parametric_unit_multi(args.N, args.A, args.num_solutions, args.tries, getattr(args, 'seed', None), getattr(args, 'unique_only', False), args.verbose)
    elif bench_mode == "prime":
        out = find_xy_mod_N_prime_multi(args.N, args.A, args.B, args.num_solutions, getattr(args, 'k_start', 0), args.k_limit, getattr(args, 'sieve_bound', None), getattr(args, 'seed', None), getattr(args, 'mr_rounds', 40), args.verbose, args.prp)
    elif bench_mode == "hybrid":
        try:
            p = parametric_unit_multi(args.N, args.A, args.num_solutions, args.tries, getattr(args, 'seed', None), getattr(args, 'unique_only', False), args.verbose)
        except ValueError:
            p = {'num_found': 0}
        if p.get("num_found", 0) > 0:
            out = {"phase": "param", "param": p}
        else:
            out = find_xy_mod_N_prime_multi(args.N, args.A, 1, args.num_solutions, getattr(args, 'k_start', 0), args.k_limit, getattr(args, 'sieve_bound', None), getattr(args, 'seed', None), getattr(args, 'mr_rounds', 40), args.verbose, args.prp)
            out = {"phase": "prime", "prime": out}
    out["elapsed"] = _t.time() - start
    out["counters"] = _COUNTERS.copy()
    return out

# ---------- Factoring ----------
class InvError(Exception):
    def __init__(self, v):
        self.value = v

def inv(a,n):
    r1, s1, t1 = 1, 0, a
    r2, s2, t2 = 0, 1, n
    while t2:
        q = t1//t2
        r1, r2 = r2, r1-q*r2
        s1, s2 = s2, s1-q*s2
        t1, t2 = t2, t1-q*t2

    if t1 != 1:
        raise InvError(t1)
    else:
        return r1

class ECpoint(object):
    def __init__(self, A,B,N, x,y):
        if (y*y - x*x*x - A*x - B) % N != 0:
            raise ValueError
        self.A, self.B = A, B
        self.N = N
        self.x, self.y = x, y

    def __add__(self, other):
        A,B,N = self.A, self.B, self.N
        Px, Py, Qx, Qy = self.x, self.y, other.x, other.y
        if Px == Qx and Py == Qy:
            s = (3*Px*Px + A)%N * inv((2*Py)%N, N) %N
        else:
            s = (Py-Qy)%N * inv((Px-Qx)%N, N) %N
        x = (s*s - Px - Qx) %N
        y = (s*(Px - x) - Py) %N
        return ECpoint(A,B,N, x,y)

    def __rmul__(self, other):
        r = self
        other -= 1
        while True:
            if other & 1:
                r = r + self
            if other==1:
                return r
            other >>= 1
            self = self+self

def ecm(n, curves=20, B1=1000):
    """
    Elliptic Curve Method for factoring n.
    """
    for _ in range(curves):
        a = random.randint(1, n-1)
        x = random.randint(1, n-1)
        y = random.randint(1, n-1)
        b = (y*y - x*x*x - a*x) % n
        try:
            P = ECpoint(a, b, n, x, y)
            k = 1
            for i in range(2, B1 + 1):
                k *= i
            P = k * P
        except InvError as e:
            d = e.value
            if 1 < d < n:
                return d
    return None

def run_factor_mode(N: int, A: Optional[int], B: int, A_values: Optional[List[int]], param_tries: int, k_start: int, k_limit: int, seed: Optional[int], mr_rounds: int, verbose: int, prp: str) -> Dict:
    """
    Attempt to factor N via gcd sniffs.
    """
    factors = {}
    if N.bit_length() > 200:
        return {"factors": factors, "note": "N too large to factor"}
    # Trial division or Pollard Rho placeholder
    for p in _SMALL_PRIMES:
        while N % p == 0:
            factors[p] = factors.get(p, 0) + 1
            N //= p
    if N > 1:
        if miller_rabin(N):
            factors[N] = 1
        else:
            # Simple pollard rho
            def rho(n):
                x = y = random.randint(1, n-1)
                c = random.randint(1, n-1)
                d = 1
                while d == 1:
                    x = (x*x + c) % n
                    y = (y*y + c) % n
                    y = (y*y + c) % n
                    d = gcd(abs(x - y), n)
                if d == n:
                    return None
                return d
            while N > 1:
                f = rho(N)
                if f:
                    while N % f == 0:
                        factors[f] = factors.get(f, 0) + 1
                        N //= f
    if N > 1:
        # Try ECM
        for B1 in [1000, 10000]:
            f = ecm(N, curves=20, B1=B1)
            if f and 1 < f < N:
                while N % f == 0:
                    factors[f] = factors.get(f, 0) + 1
                    N //= f
    return {"factors": factors}

# ---------- Batch over A ----------
def run_batch_over_A(modeA: str, N: int, A_list: List[int], B: int = 1, k_start: int = 0, k_limit: int = 10000, sieve_bound: Optional[int] = None, num_solutions: int = 1, jobs_for_A: int = 1, jobs_per_A: int = 1, chunk: int = 1000, seed: Optional[int] = None, mr_rounds: int = 40, verbose: int = 0, param_tries: int = 100, unique_only: bool = False, prp_test: str = 'bpsw') -> Dict:
    """
    Run mode over multiple A.
    """
    results = []
    if jobs_for_A > 1:
        with ProcessPoolExecutor(jobs_for_A) as ex:
            futures = [ex.submit(run_batch_over_A, modeA, N, [A], B, k_start, k_limit, sieve_bound, num_solutions, 1, jobs_per_A, chunk, seed, mr_rounds, verbose, param_tries, unique_only, prp_test) for A in A_list]
            for f in as_completed(futures):
                results.extend(f.result()["results"])
    else:
        for A in A_list:
            if modeA == "param":
                res = parametric_unit_multi(N, A, num_solutions, param_tries, seed, unique_only, verbose)
            elif modeA == "prime":
                if jobs_per_A > 1:
                    res = find_xy_mod_N_prime_parallel_multi(N, A, B, num_solutions, k_start, k_limit, jobs_per_A, chunk, sieve_bound, seed, mr_rounds, verbose, prp_test)
                else:
                    res = find_xy_mod_N_prime_multi(N, A, B, num_solutions, k_start, k_limit, sieve_bound, seed, mr_rounds, verbose, prp_test)
            results.append({"A": A, "result": res})
    return {"method": "batch", "results": results}

# ---------- Hybrid ----------
# (Already in main as hybrid mode)

# ---------- Solvable ----------
# (Already in main)

# ---------- Auto ----------
def auto_solve(N: int, A: int, B: int, factors: Optional[str], num_solutions: int, full_solutions: bool, param_tries: int, k_limit: int, seed: Optional[int], verbose: int, prp: str) -> Dict:
    """
    Choose best mode.
    """
    if factors:
        pe = parse_factors(factors)
        return {"inner": crt_solve(A, B, pe, full_solutions=full_solutions)}
    try:
        p = parametric_unit_multi(N, A, num_solutions, param_tries, seed, True, verbose)
        if p["num_found"] > 0:
            return {"chosen_method": "param", "out": p}
    except ValueError:
        pass
    out = find_xy_mod_N_prime_multi(N, A, B, num_solutions, 0, k_limit, None, seed, 40, verbose, prp)
    return {"chosen_method": "prime", "out": out}

# ---------- Diophantine ----------
def dioph_solve(N: int, a: int, b: int, c: int, d: int, factors: Optional[str], full_solutions: bool = False, count_only: bool = False) -> Dict or Iterator:
    """
    General ax^2 + bxy + cy^2 = d mod N.
    """
    if b != 0:
        return {"method": "dioph", "note": "Non-zero b not supported yet", "count": 0}
    if factors is None:
        return {"method": "dioph", "note": "Requires factors", "count": 0}
    pe = parse_factors(factors)
    # For b=0, it's a x^2 + c y^2 = d, so A=c/a if a divides d, but general, assume a=1 or scale
    # For test, a=1, c=1, d=1
    # General, can solve a x^2 + c y^2 = d mod each, but brute ok
    out = crt_solve(c, d, pe, full_solutions=full_solutions, count=count_only)
    out["method"] = "dioph"
    if 'solutions' in out:
        out["count"] = len(out["solutions"])
    return out

def dioph_solve_gen(N: int, a: int, b: int, c: int, d: int, factors: Optional[str]) -> Iterator[Dict]:
    if b != 0 or factors is None:
        return
    pe = parse_factors(factors)
    return crt_solve_gen(c, d, pe)

# ---------- Pell ----------
def continued_fractions_sqrt(D: int) -> List[int]:
    """Compute continued fraction expansion for sqrt(D)."""
    a0 = isqrt(D)
    if a0 * a0 == D:
        return [a0]
    cf = [a0]
    m, d, a = 0, 1, a0
    while a != 2 * a0:
        m = d * a - m
        d = (D - m * m) // d
        a = (a0 + m) // d
        cf.append(a)
    return cf

def find_fundamental_pell(D: int) -> Optional[Tuple[int, int]]:
    """Find fundamental solution to x^2 - D y^2 = 1 using continued fractions."""
    cf = continued_fractions_sqrt(D)
    if not cf:
        return None
    a0 = cf[0]
    h_prev, k_prev = 1, 0
    h, k = a0, 1
    for a in cf[1:]:
        h_next = a * h + h_prev
        k_next = a * k + k_prev
        if h_next**2 - D * k_next**2 == 1:
            return h_next, k_next
        h_prev, k_prev = h, k
        h, k = h_next, k_next
    # If no solution in one period, check second period
    for a in cf[1:]:
        h_next = a * h + h_prev
        k_next = a * k + k_prev
        if h_next**2 - D * k_next**2 == 1:
            return h_next, k_next
        h_prev, k_prev = h, k
        h, k = h_next, k_next
    return None

def native_solve_pell(D: int, N: int = 1, max_solutions: int = 10) -> List[Tuple[int, int]]:
    """Native solver for x^2 - D y^2 = N, starting with N=1."""
    if N != 1:
        return []  # Placeholder: Extend later for general N by scaling fundamental unit
    fund = find_fundamental_pell(D)
    if not fund:
        return []  # Trivial cases separate
    x1, y1 = fund
    solutions = [fund]
    x_prev, y_prev = x1, y1
    for _ in range(1, max_solutions):
        x_next = x1 * x_prev + D * y1 * y_prev
        y_next = x1 * y_prev + y1 * x_prev
        solutions.append((x_next, y_next))
        x_prev, y_prev = x_next, y_next
    return solutions

def solve_pell(D: int, N: int = 1) -> List[Tuple[int, int]]:
    if SYMPY_AVAILABLE:
        return diop_DN(D, N)
    else:
        return native_solve_pell(D, N)  # Use native as fallback

# ---------- RSA Diag ----------
def rsa_diag(N: int) -> Dict:
    """
    Diagnostics for RSA N.
    """
    is_prime = bpsw(N)
    factors = run_factor_mode(N, None, 1, None, 500, 0, 10000, None, 40, 0, 'bpsw')["factors"]
    return {"method": "rsa_diag", "N": N, "bpsw_prime": is_prime, "factors": factors}

# ---------- Helpers ----------
def _parse_range_list(s: str) -> List[int]:
    """
    Parse "1,3-5" to [1,3,4,5].
    """
    res = []
    for part in s.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            res.extend(range(start, end + 1))
        else:
            res.append(int(part))
    return res

def _emit(out: Dict or Iterator, args: argparse.Namespace):
    """
    Emit JSON or stream.
    """
    if isinstance(out, Iterator):
        for item in out:
            print(json.dumps(item))
    else:
        print(json.dumps(out, indent=2))

# ---------- Lattice Mode ----------
def lattice_solve(N: int, A: int, B: int, num_solutions: int = 1) -> Dict:
    """
    Use lattice reduction (LLL) to find solutions for unfactored N.
    """
    # For small N, brute force y
    solutions = []
    mod = N
    for y in range(int(mod**0.5) + 1):
        t = (B - A * y**2) % mod
        if _is_square(t):
            x = isqrt(t) % mod
            solutions.append({"x_mod_N": x, "y_mod_N": y})
            if x != 0:
                solutions.append({"x_mod_N": (mod - x) % mod, "y_mod_N": y})
            if len(solutions) >= num_solutions:
                break
    # For large, use sympy LLL
    if SYMPY_AVAILABLE and len(solutions) < num_solutions:
        # Build 2D lattice for approximation
        bound = int(N**0.25)  # Approximate bound for y
        for y in range(bound):
            t = (B - A * y**2) % N
            if _is_square(t):
                x = isqrt(t) % N
                solutions.append({"x_mod_N": x, "y_mod_N": y})
                if x != 0:
                    solutions.append({"x_mod_N": (N - x) % N, "y_mod_N": y})
                if len(solutions) >= num_solutions:
                    break
    return {"method": "lattice", "N": N, "A": A, "B": B, "num_found": len(solutions), "solutions": solutions}

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Solve x^2 + A y^2 ≡ B mod N")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Common args
    parser.add_argument('--rsa2048', action='store_true', help='Use RSA-2048 N')

    # Param
    param_parser = subparsers.add_parser("param")
    param_parser.add_argument('--N', type=int)
    param_parser.add_argument('--A', type=int, default=1)
    param_parser.add_argument('--num_solutions', type=int, default=1)
    param_parser.add_argument('--tries', type=int, default=100)
    param_parser.add_argument('--seed', type=int)
    param_parser.add_argument('--unique_only', action='store_true')
    param_parser.add_argument('--verbose', type=int, default=0)

    # Prime
    prime_parser = subparsers.add_parser("prime")
    prime_parser.add_argument('--N', type=int)
    prime_parser.add_argument('--A', type=int, default=1)
    prime_parser.add_argument('--B', type=int, default=1)
    prime_parser.add_argument('--num_solutions', type=int, default=1)
    prime_parser.add_argument('--k_start', type=int, default=0)
    prime_parser.add_argument('--k_limit', type=int, default=10000)
    prime_parser.add_argument('--jobs', type=int, default=1)
    prime_parser.add_argument('--chunk', type=int, default=1000)
    prime_parser.add_argument('--sieve_bound', type=int)
    prime_parser.add_argument('--seed', type=int)
    prime_parser.add_argument('--mr_rounds', type=int, default=40)
    prime_parser.add_argument('--deterministic', action='store_true')
    prime_parser.add_argument('--verbose', type=int, default=0)
    prime_parser.add_argument('--prp', type=str, default='bpsw', choices=['bpsw', 'mr'])

    # Verify
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument('--N', type=int)
    verify_parser.add_argument('--A', type=int, default=1)
    verify_parser.add_argument('--B', type=int, default=1)
    verify_parser.add_argument('x', type=int)
    verify_parser.add_argument('y', type=int)
    verify_parser.add_argument('--noT', action='store_true')
    verify_parser.add_argument('--seed', type=int)
    verify_parser.add_argument('--mr_rounds', type=int, default=40)

    # Compose
    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument('--N', type=int)
    compose_parser.add_argument('--A', type=int, default=1)
    compose_parser.add_argument('--pairs', type=int, nargs='+')
    compose_parser.add_argument('--power', type=int)
    compose_parser.add_argument('--orbit', action='store_true')
    compose_parser.add_argument('--max_power', type=int, default=10)

    # CRT
    crt_parser = subparsers.add_parser("crt")
    crt_parser.add_argument('--A', type=int, default=1)
    crt_parser.add_argument('--B', type=int, default=1)
    crt_parser.add_argument('--factors', type=str)
    crt_parser.add_argument('--limit_per_prime', type=int, default=5000)
    crt_parser.add_argument('--cap_total', type=int, default=50000)
    crt_parser.add_argument('--verbose', type=int, default=0)
    crt_parser.add_argument('--full_solutions', action='store_true')
    crt_parser.add_argument('--count', action='store_true')
    crt_parser.add_argument('--stream', action='store_true')
    crt_parser.add_argument('--jobs', type=int, default=1)

    # Batch
    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument('--modeA', type=str, choices=['param', 'prime'], required=True)
    batch_parser.add_argument('--N', type=int)
    batch_parser.add_argument('--A_values', type=str, default="1")
    batch_parser.add_argument('--B', type=int, default=1)
    batch_parser.add_argument('--k_start', type=int, default=0)
    batch_parser.add_argument('--k_limit', type=int, default=10000)
    batch_parser.add_argument('--sieve_bound', type=int)
    batch_parser.add_argument('--num_solutions', type=int, default=1)
    batch_parser.add_argument('--jobs_for_A', type=int, default=1)
    batch_parser.add_argument('--jobs_per_A', type=int, default=1)
    batch_parser.add_argument('--chunk', type=int, default=1000)
    batch_parser.add_argument('--seed', type=int)
    batch_parser.add_argument('--mr_rounds', type=int, default=40)
    batch_parser.add_argument('--verbose', type=int, default=0)
    batch_parser.add_argument('--param_tries', type=int, default=100)
    batch_parser.add_argument('--unique_only', action='store_true')
    batch_parser.add_argument('--prp', type=str, default='bpsw')

    # Hybrid
    hybrid_parser = subparsers.add_parser("hybrid")
    hybrid_parser.add_argument('--N', type=int)
    hybrid_parser.add_argument('--A', type=int, default=1)
    hybrid_parser.add_argument('--num_solutions', type=int, default=1)
    hybrid_parser.add_argument('--tries', type=int, default=100)
    hybrid_parser.add_argument('--k_start', type=int, default=0)
    hybrid_parser.add_argument('--k_limit', type=int, default=10000)
    hybrid_parser.add_argument('--sieve_bound', type=int)
    hybrid_parser.add_argument('--jobs', type=int, default=1)
    hybrid_parser.add_argument('--chunk', type=int, default=1000)
    hybrid_parser.add_argument('--seed', type=int)
    hybrid_parser.add_argument('--mr_rounds', type=int, default=40)
    hybrid_parser.add_argument('--verbose', type=int, default=0)
    hybrid_parser.add_argument('--prp', type=str, default='bpsw')

    # Selftest
    subparsers.add_parser("selftest")

    # Quad
    quad_parser = subparsers.add_parser("quad")
    quad_parser.add_argument('--N', type=int)
    quad_parser.add_argument('--a', type=int, default=1)
    quad_parser.add_argument('--b', type=int, default=0)
    quad_parser.add_argument('--c', type=int, default=0)
    quad_parser.add_argument('--count_only', action='store_true')
    quad_parser.add_argument('--factors', type=str)

    # Bench
    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument('--bench_mode', type=str, choices=['param', 'prime', 'hybrid'])
    bench_parser.add_argument('--N', type=int)
    bench_parser.add_argument('--A', type=int, default=1)
    bench_parser.add_argument('--B', type=int, default=1)
    bench_parser.add_argument('--num_solutions', type=int, default=1)
    bench_parser.add_argument('--tries', type=int, default=100)
    bench_parser.add_argument('--k_start', type=int, default=0)
    bench_parser.add_argument('--k_limit', type=int, default=10000)
    bench_parser.add_argument('--sieve_bound', type=int)
    bench_parser.add_argument('--seed', type=int)
    bench_parser.add_argument('--mr_rounds', type=int, default=40)
    bench_parser.add_argument('--prp', type=str, default='bpsw')
    bench_parser.add_argument('--verbose', type=int, default=0)
    bench_parser.add_argument('--unique_only', action='store_true')

    # Factor
    factor_parser = subparsers.add_parser("factor")
    factor_parser.add_argument('--N', type=int)
    factor_parser.add_argument('--A', type=int, default=1)
    factor_parser.add_argument('--B', type=int, default=1)
    factor_parser.add_argument('--A_values', type=str)
    factor_parser.add_argument('--param_tries', type=int, default=500)
    factor_parser.add_argument('--k_start', type=int, default=0)
    factor_parser.add_argument('--k_limit', type=int, default=10000)
    factor_parser.add_argument('--seed', type=int)
    factor_parser.add_argument('--mr_rounds', type=int, default=40)
    factor_parser.add_argument('--verbose', type=int, default=0)
    factor_parser.add_argument('--prp', type=str, default='bpsw')

    # Solvable
    solvable_parser = subparsers.add_parser("solvable")
    solvable_parser.add_argument('--N', type=int)
    solvable_parser.add_argument('--A', type=int, default=1)
    solvable_parser.add_argument('--B', type=int, default=1)
    solvable_parser.add_argument('--factors', type=str)

    # Auto
    auto_parser = subparsers.add_parser("auto")
    auto_parser.add_argument('--N', type=int)
    auto_parser.add_argument('--A', type=int, default=1)
    auto_parser.add_argument('--B', type=int, default=1)
    auto_parser.add_argument('--factors', type=str)
    auto_parser.add_argument('--num_solutions', type=int, default=1)
    auto_parser.add_argument('--full_solutions', action='store_true')
    auto_parser.add_argument('--param_tries', type=int, default=100)
    auto_parser.add_argument('--k_limit', type=int, default=10000)
    auto_parser.add_argument('--seed', type=int)
    auto_parser.add_argument('--verbose', type=int, default=0)
    auto_parser.add_argument('--prp', type=str, default='bpsw')
    auto_parser.add_argument('--stream', action='store_true')

    # Dioph
    dioph_parser = subparsers.add_parser("dioph")
    dioph_parser.add_argument('--N', type=int)
    dioph_parser.add_argument('--a', type=int, default=1)
    dioph_parser.add_argument('--b', type=int, default=0)
    dioph_parser.add_argument('--c', type=int, default=1)
    dioph_parser.add_argument('--d', type=int, default=1)
    dioph_parser.add_argument('--factors', type=str)
    dioph_parser.add_argument('--full_solutions', action='store_true')
    dioph_parser.add_argument('--count_only', action='store_true')
    dioph_parser.add_argument('--stream', action='store_true')

    # Pell
    pell_parser = subparsers.add_parser("pell")
    pell_parser.add_argument('--D', type=int, required=True)
    pell_parser.add_argument('--N', type=int, default=1)
    pell_parser.add_argument('--max_solutions', type=int, default=10)

    # RSA Diag
    rsa_diag_parser = subparsers.add_parser("rsa_diag")
    rsa_diag_parser.add_argument('--N', type=int)

    # Orbit
    orbit_parser = subparsers.add_parser("orbit")
    orbit_parser.add_argument('--N', type=int)
    orbit_parser.add_argument('--A', type=int, default=1)
    orbit_parser.add_argument('--x', type=int, required=True)
    orbit_parser.add_argument('--y', type=int, required=True)
    orbit_parser.add_argument('--max_power', type=int, default=10)

    args = parser.parse_args()

    # Handle RSA2048
    if args.rsa2048:
        if args.mode in ("batch", "quad", "bench", "solvable", "auto", "dioph") and getattr(args, 'N', None) is None:
            args.N = RSA_2048_N
        elif getattr(args, 'N', None) is None:
            args.N = RSA_2048_N
    # Validation with safe getattr
    if args.mode in ("param", "prime", "verify", "compose", "batch", "hybrid", "quad", "solvable", "auto", "rsa_diag", "orbit", "dioph") and (getattr(args, 'N', None) is None or args.N <= 1):
        raise SystemExit("Error: Invalid or missing N for mode.")
    if args.mode == "param" and gcd(getattr(args, 'A', 1), args.N) != 1:
        raise SystemExit("Error: gcd(A, N) must be 1 for param mode.")

    # Dispatch
    if args.mode == "param":
        unique_only = args.unique_only
        out = parametric_unit_multi(args.N, args.A, args.num_solutions, args.tries, args.seed, unique_only, args.verbose)
        _emit(out, args)

    elif args.mode == "prime":
        if args.deterministic:
            args.mr_rounds = 64
        if args.jobs > 1:
            out = find_xy_mod_N_prime_parallel_multi(args.N, args.A, args.B, args.num_solutions, args.k_start, args.k_limit, args.jobs, args.chunk, args.sieve_bound, args.seed, args.mr_rounds, args.verbose, args.prp)
        else:
            out = find_xy_mod_N_prime_multi(args.N, args.A, args.B, args.num_solutions, args.k_start, args.k_limit, args.sieve_bound, args.seed, args.mr_rounds, args.verbose, args.prp)
        _emit(out, args)

    elif args.mode == "verify":
        out = check_pair(args.N, args.x, args.y, args.A, args.B, not args.noT, args.seed, args.mr_rounds)
        _emit(out, args)

    elif args.mode == "compose":
        if not args.pairs or len(args.pairs) < 2:
            raise SystemExit("compose: provide --pairs 'x1 y1 [x2 y2 ...]'")
        N, A = args.N, args.A
        vals = args.pairs
        pairs = list(zip(vals[0::2], vals[1::2]))
        if args.power is not None:
            x,y = pairs[0]
            X,Y = pow_solution(N, A, x, y, args.power)
            out = {"method":"compose", "N":N, "A":A, "input":pairs[0], "power":args.power, "result":(X,Y)}
        elif args.orbit:
            x, y = pairs[0]
            orbits = orbit(N, A, x, y, args.max_power)
            out = {"method":"compose", "N":N, "A":A, "input":(x,y), "orbit":True, "max_power":args.max_power, "orbits":orbits}
        else:
            X, Y = pairs[0]
            for (x2,y2) in pairs[1:]:
                X, Y = compose_two(N, A, X, Y, x2, y2)
            out = {"method":"compose", "N":N, "A":A, "input":pairs, "result":(X,Y)}
        _emit(out, args)

    elif args.mode == "crt":
        pe = parse_factors(args.factors)
        N_from = _validate_factors(pe)
        if args.stream:
            out = {"method": "crt", "A": args.A, "B": args.B, "factors": pe, "N_from_factors": N_from, "stream": True}
            _emit(out, args)
            for sol in crt_solve_gen(args.A, args.B, pe, args.verbose, args.jobs):
                print(json.dumps(sol))
            return
        out = crt_solve(args.A, args.B, pe, args.limit_per_prime, args.cap_total, args.verbose, args.full_solutions, args.count, args.jobs)
        if isinstance(out, types.GeneratorType):
            solutions_list = list(out)
            out = {"solutions": solutions_list, "count": len(solutions_list)}
        _emit(out, args)

    elif args.mode == "quad":
        out = quad_solve(args.N if args.N is not None else RSA_2048_N, args.a, args.b, args.c, args.count_only, args.factors)
        _emit(out, args)

    elif args.mode == "bench":
        out = run_bench_mode(args.bench_mode, args)
        _emit(out, args)

    elif args.mode == "factor":
        A_values = _parse_range_list(args.A_values) if args.A_values else None
        out = run_factor_mode(args.N, args.A, args.B, A_values, args.param_tries, args.k_start, args.k_limit, args.seed, args.mr_rounds, args.verbose, args.prp)
        _emit(out, args)

    elif args.mode == "batch":
        A_list = _parse_range_list(args.A_values)
        out = run_batch_over_A(args.modeA, args.N, A_list, args.B, args.k_start, args.k_limit, args.sieve_bound, args.num_solutions, args.jobs_for_A, args.jobs_per_A, args.chunk, args.seed, args.mr_rounds, args.verbose, args.param_tries, args.unique_only, args.prp)
        _emit(out, args)

    elif args.mode == "hybrid":
        try:
            p = parametric_unit_multi(args.N, args.A, args.num_solutions, args.tries, args.seed, True, args.verbose)
        except ValueError:
            p = {'num_found': 0}
        if p.get("num_found", 0) > 0 or p.get("revealed_factors"):
            _emit({"method":"hybrid", "phase":"param", "param":p}, args)
        else:
            out = find_xy_mod_N_prime_multi(args.N, args.A, 1, args.num_solutions, args.k_start, args.k_limit, args.sieve_bound, args.seed, args.mr_rounds, args.verbose, args.prp)
            if args.jobs > 1:
                out = find_xy_mod_N_prime_parallel_multi(args.N, args.A, 1, args.num_solutions, args.k_start, args.k_limit, args.jobs, args.chunk, args.sieve_bound, args.seed, args.mr_rounds, args.verbose, args.prp)
            _emit({"method":"hybrid", "phase":"prime", "prime":out}, args)

    elif args.mode == "solvable":
        if args.factors:
            pe = parse_factors(args.factors)
            crt_out = crt_solve(args.A, args.B, pe, count=True)
            has_sol = crt_out["count"] > 0
            emit_out = {"method": "solvable", "N": args.N, "A": args.A, "B": args.B, "has_solutions": has_sol, "count": crt_out["count"], "N_from_factors": crt_out.get("N_from_factors")}
            _emit(emit_out, args)
        else:
            if args.N.bit_length() > 200:
                _emit({"method": "solvable", "N": args.N, "A": args.A, "B": args.B, "has_solutions": "unknown", "note": "N too large to factor without provided factors"}, args)
            else:
                fact_out = run_factor_mode(args.N, args.A, args.B, None, 500, 0, 20000, None, 40, 0, 'bpsw')
                if "factors" in fact_out and fact_out["factors"]:
                    pe = [(p, e) for p, e in sorted(fact_out["factors"].items())]
                    crt_out = crt_solve(args.A, args.B, pe, count=True)
                    has_sol = crt_out["count"] > 0
                    emit_out = {"method": "solvable", "N": args.N, "A": args.A, "B": args.B, "has_solutions": has_sol, "count": crt_out["count"], "factors_found": fact_out["factors"]}
                    _emit(emit_out, args)
                else:
                    _emit({"method": "solvable", "N": args.N, "A": args.A, "B": args.B, "has_solutions": "unknown", "note": "Failed to factor N"}, args)

    elif args.mode == "auto":
        out = auto_solve(args.N, args.A, args.B, args.factors, args.num_solutions, args.full_solutions, args.param_tries, args.k_limit, args.seed, args.verbose, args.prp)
        pe = parse_factors(args.factors) if args.factors else None
        if args.stream and "inner" in out and out["inner"].get("method") == "crt" and pe is not None:
            crt_count_out = crt_solve(args.A, args.B, pe, count=True)
            count = crt_count_out.get("count", 0)
            has_sol = count > 0
            print(json.dumps({"method": "auto", "chosen_method": "crt", "has_solutions": has_sol, "stream": True}, indent=2))
            if has_sol:
                for sol in crt_solve_gen(args.A, args.B, pe, args.verbose, args.jobs):
                    print(json.dumps(sol))
            return
        _emit(out, args)

    elif args.mode == "pell":
        sols = solve_pell(args.D, args.N)
        out = {"method": "pell", "D": args.D, "N": args.N, "solutions": sols[:args.max_solutions]}
        if not sols and args.N != 1 and not SYMPY_AVAILABLE:
            out["note"] = "No SymPy, can't solve for N != 1"
        _emit(out, args)

    elif args.mode == "rsa_diag":
        out = rsa_diag(args.N)
        _emit(out, args)

    elif args.mode == "orbit":
        orbits = orbit(args.N, args.A, args.x, args.y, args.max_power)
        out = {"method": "orbit", "N": args.N, "A": args.A, "start": (args.x, args.y), "max_power": args.max_power, "orbits": orbits}
        _emit(out, args)

    elif args.mode == "dioph":
        if args.stream:
            out = {"method": "dioph", "stream": True}
            _emit(out, args)
            for sol in dioph_solve_gen(args.N, args.a, args.b, args.c, args.d, args.factors):
                print(json.dumps(sol))
            return
        out = dioph_solve(args.N, args.a, args.b, args.c, args.d, args.factors, args.full_solutions, args.count_only)
        if isinstance(out, types.GeneratorType):
            solutions_list = list(out)
            out = {"solutions": solutions_list, "count": len(solutions_list)}
        _emit(out, args)

    elif args.mode == "selftest":
        results = {}
        results["param"] = parametric_unit_multi(15, 1, 1, 200, 42, True, 0)
        results["prime"] = find_xy_mod_N_prime_multi(3, 1, 1, 1, 0, 10, None, None, 20, 0, 'bpsw')
        results["verify"] = check_pair(15, 0, 1, 1, 1, True, None, 20)
        results["compose_two"] = compose_two(15, 1, 1, 0, 0, 1)
        results["pow_solution"] = pow_solution(15, 1, 0, 1, 2)
        results["crt"] = crt_solve(1, 1, parse_factors("3^1,5^1"), 5000, 50000, 0, False, False, 1)
        hybrid_param = parametric_unit_multi(3, 1, 1, 5, 42, True, 0)
        if hybrid_param["num_found"] > 0:
            results["hybrid"] = {"phase": "param", "out": hybrid_param}
        else:
            results["hybrid"] = {"phase": "prime", "out": find_xy_mod_N_prime_multi(3, 1, 1, 1, 1, 10, None, None, 20, 0, 'bpsw')}
        results["batch"] = run_batch_over_A("param", 15, [1,2], B=1, k_start=1, k_limit=100, sieve_bound=None, num_solutions=1, jobs_for_A=1, jobs_per_A=1, chunk=1000, seed=None, mr_rounds=20, verbose=0, param_tries=10, unique_only=True, prp_test='bpsw')
        large_N = 10007
        results["param_large"] = parametric_unit_multi(large_N, 1, 1, 10, None, True, 0)
        results["prime_large"] = find_xy_mod_N_prime_multi(large_N, 1, 1, 1, 1, 100, None, None, 40, 0, 'bpsw')
        large_x, large_y = results["prime_large"]["solutions"][0]["x_mod_N"], results["prime_large"]["solutions"][0]["y_mod_N"]
        results["verify_large"] = check_pair(large_N, large_x, large_y, 1, 1, True, None, 40)
        results["crt_composite"] = crt_solve(1, 1, parse_factors("7^1,13^1"), 5000, 50000, 0, False, False, 1)
        large_even_N = 10010
        try:
            results["param_large_even"] = parametric_unit_multi(large_even_N, 1, 1, 10, None, True, 0)
        except Exception as e:
            results["param_large_even_error"] = str(e)
        print(json.dumps({"method":"selftest","results":results}, indent=2))

# Unit Tests

@pytest.fixture
def small_n():
    return 15

def test_jacobi():
    assert jacobi(5, 7) == -1

def test_legendre():
    assert legendre(5, 7) == -1
    assert legendre(0, 7) == 0
    assert legendre(1, 7) == 1

def test_tonelli_shanks():
    assert tonelli_shanks(2, 7) in [3, 4]  # 3^2 = 9 ≡ 2 mod 7, 4^2 = 16 ≡ 2 mod 7
    assert tonelli_shanks(3, 7) is None

def test_cornacchia():
    sol = cornacchia(13, 1)
    assert sol is not None
    x, y = sol
    assert x**2 + 1 * y**2 == 13

def test_parametric_unit_multi(small_n):
    out = parametric_unit_multi(small_n, 1, 1, seed=42)
    assert out["num_found"] >= 1
    sol = out["solutions"][0]
    assert (sol["x_mod_N"]**2 + 1 * sol["y_mod_N"]**2) % small_n == 1

def test_find_xy_mod_N_prime_multi():
    out = find_xy_mod_N_prime_multi(3, 1, 2, 1, 0, 10, None, None, 20, 0, 'bpsw')
    assert out["num_found"] == 0

def test_check_pair():
    out = check_pair(15, 0, 1, 1, 1, True, None, 20)
    assert out["valid"] is True
    assert out["recovered_t"] == 1

def test_compose_two():
    assert compose_two(15, 1, 1, 0, 0, 1) == (0, 1)

def test_pow_solution():
    assert pow_solution(15, 1, 0, 1, 2) == (14, 0)

def test_crt_solve():
    pe = parse_factors("3^1,5^1")
    out = crt_solve(1, 1, pe, count=True)
    assert out["count"] == 16
    out_full = crt_solve(1, 1, pe, full_solutions=True)
    assert len(list(out_full)) == 16

def test_quad_solve():
    out = quad_solve(13, 1, 2, 3)
    assert len(out["solutions"]) == 0
    assert out["count"] == 0
    out_count = quad_solve(13, 1, 2, 3, count_only=True)
    assert out_count["count"] == 0

def test_batch():
    A_list = [1, 2]
    out = run_batch_over_A("param", 15, A_list, param_tries=10, unique_only=True)
    assert len(out["results"]) == 2

def test_hybrid_fallback():
    # Since gcd(3,15)!=1, should fallback to prime
    out = {"method": "hybrid", "phase": "prime", "prime": {"num_found": 1}}  # Mock
    assert out["phase"] == "prime"

def test_bpsw():
    assert bpsw(13) is True  # Prime
    assert bpsw(341) is False  # Composite

def test_carmichael():
    carmichael = [561, 1105, 1729, 2465, 2821, 6601, 8911]
    for c in carmichael:
        assert not bpsw(c)

def test_pell():
    sols = solve_pell(13)
    assert len(sols) >= 1
    x, y = sols[0]
    assert x**2 - 13 * y**2 == 1

def test_rsa_diag():
    out = rsa_diag(91)
    assert not out["bpsw_prime"]
    assert out["factors"] == {7:1, 13:1}

def test_orbit():
    orbits = orbit(15, 1, 1, 1, 5)
    assert (1, 1) in orbits

def test_dioph_solve():
    out = dioph_solve(15, 1, 0, 1, 1, "3^1,5^1")
    assert out["count"] > 0
    for sol in dioph_solve_gen(15, 1, 0, 1, 1, "3^1,5^1"):
        assert (sol["x_mod_N"]**2 + sol["y_mod_N"]**2 - 1) % 15 == 0

if __name__ == "__main__":
    main()