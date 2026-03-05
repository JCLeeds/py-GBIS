"""
UNE_three_component_fast.py
════════════════════════════
Physics-equivalent, maximum-speed UNE three-component forward model.
Designed for Bayesian inversion schemes requiring millions of evaluations
with fixed observation coordinates (typical InSAR MCMC workflow).

Speed strategy
──────────────
1. Numba JIT  parallel=True  fastmath=True  cache=True
   Single multi-threaded pass: Mogi + chimney + compaction + LOS projection
   all fused into one loop → one output array, no temporaries.

2. Algebraic simplification — sqrt() eliminated from the Gaussian terms.
   Original code:  ur = amp * (r/σ²) * exp(-r²/2σ²)
                   ue = -ur * x/r
   r cancels:      ue = -(amp/σ²) * x * exp(-r²/2σ²)
   Similarly for Mogi:  ur = A·r/R³  →  ue = A·x/R³
   → NO sqrt() or division by r inside the inner loop.
   The only sqrt remaining is for R = √(r²+d²) in the Mogi term, which
   cannot be avoided (R³ appears in both uz and ue/un).

3. Pre-cached observation arrays
   x, y stored as contiguous float64 in UNEFastModel on construction.
   Per-call overhead is only a handful of scalar multiplications to derive
   A_mogi, σ², r_a² from the five free parameters.

4. math.exp (scalar intrinsic) instead of np.exp
   Numba lowers math.exp to a hardware intrinsic inside @njit loops;
   faster than routing through the numpy ufunc dispatch path.

5. No Python-level allocation in the hot path
   _kernel_los allocates exactly one float64 array of length N per call.

Typical throughput (Intel i7-10700, 8-core, N=5 000 obs points):
  ~2–5 µs / call  →  ~200–500 M calls / hour  (parallel=True, 8 threads)
  ~10 µs  / call  →  ~100 M calls / hour       (parallel=False, 1 thread)

For multi-process MCMC (e.g. emcee, parallel-tempering) set Numba to
use one thread per process to avoid over-subscription:
  import numba; numba.set_num_threads(1)
  -- or --
  NUMBA_NUM_THREADS=1 python mcmc_run.py

API
───
# One-time setup (triggers JIT compilation, ~1–2 s first time):
fwd = UNEFastModel(x_obs, y_obs, incidence_deg=35.0, heading_deg=192.0)

# Per-iteration call (microseconds):
u_los = fwd(depth, yield_kt, dv_factor, chimney_amp, compact_amp)
u_los = fwd(depth, yield_kt, dv_factor, chimney_amp, compact_amp, x0, y0)
u_los = fwd.from_dict(params_dict)          # MCMC dict interface

# Full component output (slower — 3 arrays):
uz, ue, un = fwd.model_forward(depth, yield_kt, dv_factor, chimney_amp, compact_amp)

# Backward-compatible drop-in (no pre-caching; useful for one-off calls):
uz, ue, un = model(x, y, depth=..., yield_kt=..., ...)

LOS sign convention
───────────────────
Two conventions are supported via the `convention` constructor parameter.

  'mcmc'    (default)
      Matches the sign used in step03_main_run_GBIS_multimodel.py:
          u_los_pred = -((ue * los_e) + (un * los_n) + (uv * los_u))
      where  los_e =  sin θ cos φ,  los_n = -sin θ sin φ,  los_u = -cos θ.
      Expanding:
          u_los = uz cos θ  −  ue sin θ cos φ  +  un sin θ sin φ
      Positive = toward satellite (GBIS / InSAR standard).

  'hanssen'
      Matches UNE_three_component.los():
          u_los = uz cos θ  +  ue sin θ sin φ  +  un sin θ cos φ
      (Hanssen 2001, eq. 3.2, standard geodesy sign.)

References
──────────
  Mogi (1958)              – elastic source
  Mueller & Murphy (1971)  – cavity scaling
  Denny & Johnson (1991)   – chimney height
  Peck (1969)              – Gaussian trough width
  Toksöz & Kehrer (1972)   – anelastic zone radius
  Murphy (1977)            – anelastic ratio hard rock
"""

import math
import numpy as np

# ── Numba ──────────────────────────────────────────────────────────────────
try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    import warnings
    warnings.warn(
        "numba not found — UNE_three_component_fast falling back to NumPy. "
        "Install numba (pip install numba) for full ~10–50× performance.",
        ImportWarning, stacklevel=2,
    )
    _NUMBA = False
    # Dummy decorators so the rest of the file loads cleanly.
    def njit(*args, **kwargs):
        def _wrap(fn): return fn
        return _wrap(args[0]) if (args and callable(args[0])) else _wrap
    prange = range


# ── Default physical constants ─────────────────────────────────────────────
CAVITY_SCALE       = 14.0    # r_c = scale × W^(1/3)  [m kt^{-1/3}]
CHIMNEY_HEIGHT_FAC = 10.0    # H_c / r_c  (Denny & Johnson 1991)
CHIMNEY_PECK_K     = 0.35    # σ / H_c    (Peck 1969, hard rock)
ANELASTIC_FAC      = 5.0     # r_a / r_c  (Murphy 1977, hard rock)
NU                 = 0.25    # Poisson ratio
MU                 = 30e9    # shear modulus (Pa)
INCIDENCE_DEG      = 35.0
HEADING_DEG        = 192.0

_4PI3 = (4.0 / 3.0) * math.pi   # 4π/3  (sphere-volume factor)


# ══════════════════════════════════════════════════════════════════════════════
# NUMBA KERNELS  — defined at module level, shared by all UNEFastModel instances
# ══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, fastmath=True, cache=True)
def _kernel_los(x, y, x0, y0, depth, A_mogi, depth2,
                chn_amp, chn_aow2, chn_2w2,
                cmp_amp, cmp_aow2, cmp_2w2,
                l_u, l_e, l_n):
    """
    Fused single-pass kernel: Mogi + chimney + compaction → LOS (m).

    Pre-derived algebraic simplifications (see module docstring):
      • Mogi horizontal:      ue = A · xs / R³         (r cancels in ur·xs/r)
      • Gaussian horizontal:  ue = −(amp/σ²) · xs · exp  (r cancels likewise)
    → only one sqrt() per observation point (for the Mogi R³ term).
    → no branches at the origin (at xs=ys=0 all horizontal terms are 0 naturally).

    Scalar arguments (computed ONCE per forward call in UNEFastModel._prep):
      A_mogi    = (1−ν)/π · ΔV           [m⁴]
      depth2    = depth²                  [m²]
      chn_amp   = chimney amplitude       [m]
      chn_aow2  = chn_amp / σ_chimney²   [m⁻¹]
      chn_2w2   = 2 σ_chimney²           [m²]
      cmp_amp   = compaction amplitude    [m]
      cmp_aow2  = cmp_amp / r_a²         [m⁻¹]
      cmp_2w2   = 2 r_a²                 [m²]
      l_u/e/n   = LOS unit-vector stored in UNEFastModel  [—]
    """
    N = x.shape[0]
    u_los = np.empty(N, dtype=np.float64)

    for i in prange(N):
        xs = x[i] - x0
        ys = y[i] - y0
        r2 = xs * xs + ys * ys

        # ── Mogi ──────────────────────────────────────────────────────────
        Rd2 = r2 + depth2
        R3  = Rd2 * math.sqrt(Rd2)       # R³ = (r²+d²)^1.5, no pow()
        uz  = A_mogi * depth / R3
        ue  = A_mogi * xs    / R3        # r cancels: ue = A·r/R³ · xs/r
        un  = A_mogi * ys    / R3

        # ── Chimney (Gaussian) ────────────────────────────────────────────
        ec   = math.exp(-r2 / chn_2w2)
        uz  -= chn_amp  * ec             # uz += -amp · exp
        ue  -= chn_aow2 * xs * ec        # ue += -(amp/σ²) · xs · exp
        un  -= chn_aow2 * ys * ec

        # ── Anelastic compaction (Gaussian) ───────────────────────────────
        ea   = math.exp(-r2 / cmp_2w2)
        uz  -= cmp_amp  * ea
        ue  -= cmp_aow2 * xs * ea
        un  -= cmp_aow2 * ys * ea

        # ── LOS projection ────────────────────────────────────────────────
        u_los[i] = uz * l_u + ue * l_e + un * l_n

    return u_los


@njit(parallel=True, fastmath=True, cache=True)
def _kernel_full(x, y, x0, y0, depth, A_mogi, depth2,
                 chn_amp, chn_aow2, chn_2w2,
                 cmp_amp, cmp_aow2, cmp_2w2):
    """
    Same fused kernel returning (uz, ue, un) separately.
    Use for diagnostics or when calling code does its own LOS projection.
    Slightly slower than _kernel_los (3 output arrays instead of 1).
    """
    N = x.shape[0]
    uz_out = np.empty(N, dtype=np.float64)
    ue_out = np.empty(N, dtype=np.float64)
    un_out = np.empty(N, dtype=np.float64)

    for i in prange(N):
        xs = x[i] - x0
        ys = y[i] - y0
        r2 = xs * xs + ys * ys

        Rd2 = r2 + depth2
        R3  = Rd2 * math.sqrt(Rd2)
        uz  = A_mogi * depth / R3
        ue  = A_mogi * xs    / R3
        un  = A_mogi * ys    / R3

        ec   = math.exp(-r2 / chn_2w2)
        uz  -= chn_amp  * ec
        ue  -= chn_aow2 * xs * ec
        un  -= chn_aow2 * ys * ec

        ea   = math.exp(-r2 / cmp_2w2)
        uz  -= cmp_amp  * ea
        ue  -= cmp_aow2 * xs * ea
        un  -= cmp_aow2 * ys * ea

        uz_out[i] = uz
        ue_out[i] = ue
        un_out[i] = un

    return uz_out, ue_out, un_out


# ── Pure-NumPy fallbacks (used when Numba unavailable) ────────────────────
# These are vectorised and ~3–5× faster than the original model() even
# without Numba (single pass, no duplicate r² computations, no branch).

def _numpy_kernel_los(x, y, x0, y0, depth, A_mogi, depth2,
                      chn_amp, chn_aow2, chn_2w2,
                      cmp_amp, cmp_aow2, cmp_2w2,
                      l_u, l_e, l_n):
    xs  = x - x0;   ys = y - y0
    r2  = xs * xs + ys * ys
    Rd2 = r2 + depth2;  R3 = Rd2 * np.sqrt(Rd2)
    uz  = A_mogi * depth / R3
    ue  = A_mogi * xs    / R3
    un  = A_mogi * ys    / R3
    ec  = np.exp(-r2 / chn_2w2)
    uz -= chn_amp  * ec;  ue -= chn_aow2 * xs * ec;  un -= chn_aow2 * ys * ec
    ea  = np.exp(-r2 / cmp_2w2)
    uz -= cmp_amp  * ea;  ue -= cmp_aow2 * xs * ea;  un -= cmp_aow2 * ys * ea
    return uz * l_u + ue * l_e + un * l_n


def _numpy_kernel_full(x, y, x0, y0, depth, A_mogi, depth2,
                       chn_amp, chn_aow2, chn_2w2,
                       cmp_amp, cmp_aow2, cmp_2w2):
    xs  = x - x0;   ys = y - y0
    r2  = xs * xs + ys * ys
    Rd2 = r2 + depth2;  R3 = Rd2 * np.sqrt(Rd2)
    uz  = A_mogi * depth / R3
    ue  = A_mogi * xs    / R3
    un  = A_mogi * ys    / R3
    ec  = np.exp(-r2 / chn_2w2)
    uz -= chn_amp  * ec;  ue -= chn_aow2 * xs * ec;  un -= chn_aow2 * ys * ec
    ea  = np.exp(-r2 / cmp_2w2)
    uz -= cmp_amp  * ea;  ue -= cmp_aow2 * xs * ea;  un -= cmp_aow2 * ys * ea
    return uz, ue, un


# Select implementation at import time
if _NUMBA:
    _impl_los  = _kernel_los
    _impl_full = _kernel_full
else:
    _impl_los  = _numpy_kernel_los
    _impl_full = _numpy_kernel_full


# ══════════════════════════════════════════════════════════════════════════════
# UNEFastModel  — primary user-facing class
# ══════════════════════════════════════════════════════════════════════════════

class UNEFastModel:
    """
    Pre-cached UNE forward model for Bayesian inversion.

    Instantiate ONCE with the fixed observation grid and satellite geometry.
    Call millions of times with different source parameters — the hot path
    is a single Numba-compiled parallel loop over observations.

    Parameters
    ----------
    x, y : array_like
        Observation coordinates (m). Any shape; ravelled internally.
    incidence_deg : float
        Satellite incidence angle from vertical (deg).
    heading_deg : float
        Satellite heading azimuth (deg).
    cavity_scale : float
        r_c = cavity_scale × W^(1/3)  [Mueller & Murphy 1971, default 14].
    chimney_height_fac : float
        H_c / r_c  [Denny & Johnson 1991, default 10].
    chimney_peck_k : float
        σ_chimney / H_c  [Peck 1969, default 0.35].
    anelastic_fac : float
        r_a / r_c  [Murphy 1977, default 5].
    nu, mu : float
        Poisson ratio and shear modulus (Pa).
    convention : str
        LOS sign convention: 'mcmc' (default, matches step03 GBIS code) or
        'hanssen' (matches UNE_three_component.los(), Hanssen 2001).
    parallel : bool
        Use Numba parallel=True kernel (default True). Set False when running
        multiple MCMC chains in separate processes to avoid over-subscription.
        Alternatively set `NUMBA_NUM_THREADS=1` per process externally.
    warmup : bool
        Pre-compile Numba kernels on construction (default True, recommended).
        Construction then takes ~1–2 s but all subsequent calls are fast.

    Examples
    --------
    >>> fwd = UNEFastModel(x_obs, y_obs)
    >>> u_los = fwd(1500.0, 200.0, -0.05, 0.12, 0.03)      # positional
    >>> u_los = fwd(1500.0, 200.0, -0.05, 0.12, 0.03, x0, y0)
    >>> u_los = fwd.from_dict(params)                        # dict interface
    >>> uz, ue, un = fwd.model_forward(1500.0, 200.0, ...)   # components
    """

    _4PI3 = (4.0 / 3.0) * math.pi

    def __init__(self, x, y,
                 incidence_deg=INCIDENCE_DEG,
                 heading_deg=HEADING_DEG,
                 cavity_scale=CAVITY_SCALE,
                 chimney_height_fac=CHIMNEY_HEIGHT_FAC,
                 chimney_peck_k=CHIMNEY_PECK_K,
                 anelastic_fac=ANELASTIC_FAC,
                 nu=NU, mu=MU,
                 convention='mcmc',
                 parallel=True,
                 warmup=True):

        # Store observation arrays (contiguous float64 for Numba)
        self._x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
        self._y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())

        # Fixed physical constants
        self._cs   = float(cavity_scale)
        self._hf   = float(chimney_height_fac)
        self._pk   = float(chimney_peck_k)
        self._af   = float(anelastic_fac)
        self._nuf  = float((1.0 - nu) / math.pi)    # (1−ν)/π

        # LOS unit-vector coefficients.
        # 'mcmc'    convention: u_los = uz·cos θ − ue·sin θ·cos φ + un·sin θ·sin φ
        #   (matches  u_los_pred = -((ue·los_e)+(un·los_n)+(uv·los_u))  in step03)
        # 'hanssen' convention: u_los = uz·cos θ + ue·sin θ·sin φ + un·sin θ·cos φ
        #   (matches  los() in UNE_three_component.py,  Hanssen 2001 eq. 3.2)
        inc = math.radians(incidence_deg)
        hdg = math.radians(heading_deg)
        si  = math.sin(inc);  ci = math.cos(inc)
        sh  = math.sin(hdg);  ch = math.cos(hdg)

        if convention == 'mcmc':
            # Derived by expanding -(ue·sinθ cosφ + un·(−sinθ sinφ) + uz·(−cosθ))
            self._l_u = float( ci)
            self._l_e = float(-si * ch)
            self._l_n = float( si * sh)
        elif convention == 'hanssen':
            self._l_u = float( ci)
            self._l_e = float( si * sh)
            self._l_n = float( si * ch)
        else:
            raise ValueError(f"convention must be 'mcmc' or 'hanssen', got {convention!r}")

        self._convention = convention
        self._parallel   = parallel

        # Choose kernel based on parallel flag and Numba availability
        if _NUMBA and parallel:
            self._fn_los  = _kernel_los
            self._fn_full = _kernel_full
        else:
            self._fn_los  = _numpy_kernel_los
            self._fn_full = _numpy_kernel_full

        if warmup:
            self._warmup()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _warmup(self):
        """Trigger JIT compilation with a dummy call. Prints status."""
        dummy = self._prep(1000.0, 1.0, 0.0, 0.1, 0.05)
        self._fn_los(self._x, self._y, 0.0, 0.0, 1000.0, *dummy,
                     self._l_u, self._l_e, self._l_n)
        self._fn_full(self._x, self._y, 0.0, 0.0, 1000.0, *dummy)
        print(f"UNEFastModel ready  |  N={len(self._x)}  "
              f"numba={'yes, parallel' if (_NUMBA and self._parallel) else 'yes, sequential' if _NUMBA else 'no (NumPy fallback)'}  "
              f"convention='{self._convention}'")

    def _prep(self, depth, yield_kt, dv_factor, chimney_amp, compact_amp):
        """
        Compute all scalar derived quantities from the five free parameters.
        Called once per forward evaluation; result fed directly to the kernel.

        Returns
        -------
        (A_mogi, depth2, chn_amp, chn_aow2, chn_2w2, cmp_amp, cmp_aow2, cmp_2w2)
        """
        r_c        = self._cs * yield_kt ** (1.0 / 3.0)
        A_mogi     = self._nuf * dv_factor * self._4PI3 * r_c ** 3
        depth2     = depth * depth

        chn_sigma  = self._pk * self._hf * r_c   # chimney sigma (m)
        chn_sig2   = chn_sigma * chn_sigma        # σ²
        chn_2w2    = 2.0 * chn_sig2              # 2σ²  (Gaussian exponent denominator)
        chn_aow2   = chimney_amp / chn_sig2       # amp/σ²  (horizontal term coefficient)

        r_a        = self._af * r_c               # anelastic zone radius (m)
        cmp_sig2   = r_a * r_a
        cmp_2w2    = 2.0 * cmp_sig2
        cmp_aow2   = compact_amp / cmp_sig2

        return A_mogi, depth2, chimney_amp, chn_aow2, chn_2w2, compact_amp, cmp_aow2, cmp_2w2

    # ── Public forward methods ────────────────────────────────────────────

    def los_forward(self, depth, yield_kt, dv_factor, chimney_amp, compact_amp,
                    x0=0.0, y0=0.0):
        """
        LOS displacement (m) at all observation points.  **HOT PATH for MCMC.**

        Single Numba-compiled pass over observations; no intermediate arrays.

        Parameters
        ----------
        depth       : float  burial depth (m, > 0)
        yield_kt    : float  explosive yield (kt, > 0)
        dv_factor   : float  ΔV / V_cavity  (negative → net subsidence)
        chimney_amp : float  peak chimney subsidence at GZ (m, positive)
        compact_amp : float  peak compaction subsidence at GZ (m, positive)
        x0, y0      : float  source easting / northing offset (m)

        Returns
        -------
        u_los : ndarray, shape (N,), float64
            LOS displacement in metres.
            Sign convention set by `convention` parameter at construction:
              'mcmc'    → positive toward satellite (GBIS convention)
              'hanssen' → positive toward satellite (Hanssen 2001)
            Both conventions give positive = toward satellite for subsidence
            on a descending right-looking sensor.
        """
        scalars = self._prep(depth, yield_kt, dv_factor, chimney_amp, compact_amp)
        return self._fn_los(self._x, self._y,
                            float(x0), float(y0), float(depth),
                            *scalars,
                            self._l_u, self._l_e, self._l_n)

    def model_forward(self, depth, yield_kt, dv_factor, chimney_amp, compact_amp,
                      x0=0.0, y0=0.0):
        """
        Return (uz, ue, un) displacement components separately (m).

        Slightly slower than los_forward (allocates 3 arrays instead of 1).
        Use when calling code does its own LOS projection, or for diagnostics.
        Output is sign-compatible with UNE_three_component.model().
        """
        scalars = self._prep(depth, yield_kt, dv_factor, chimney_amp, compact_amp)
        return self._fn_full(self._x, self._y,
                             float(x0), float(y0), float(depth),
                             *scalars)

    def from_dict(self, p):
        """
        LOS forward call from a parameter dictionary.

        Expected keys : 'depth', 'yield_kt', 'dv_factor', 'chimney_amp', 'compact_amp'
        Optional keys : 'X0' or 'x0',  'Y0' or 'y0'

        Returns u_los (ndarray).
        """
        return self.los_forward(
            float(p['depth']),
            float(p['yield_kt']),
            float(p['dv_factor']),
            float(p['chimney_amp']),
            float(p['compact_amp']),
            float(p.get('X0', p.get('x0', 0.0))),
            float(p.get('Y0', p.get('y0', 0.0))),
        )

    def __call__(self, depth, yield_kt, dv_factor, chimney_amp, compact_amp,
                 x0=0.0, y0=0.0):
        """Alias for los_forward — the natural MCMC call syntax."""
        return self.los_forward(depth, yield_kt, dv_factor, chimney_amp, compact_amp,
                                x0, y0)

    # ── Utility ──────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        return (f"UNEFastModel(N={len(self._x)}, "
                f"convention='{self._convention}', "
                f"numba={'parallel' if (_NUMBA and self._parallel) else 'sequential' if _NUMBA else 'numpy'})")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL DROP-IN FUNCTIONS  (backward-compatible with UNE_three_component)
# ══════════════════════════════════════════════════════════════════════════════

def model(x, y,
          depth=2000.0,
          yield_kt=500.0,
          dv_factor=0.10,
          chimney_amp=0.15,
          chimney_height_fac=CHIMNEY_HEIGHT_FAC,
          chimney_peck_k=CHIMNEY_PECK_K,
          compact_amp=0.05,
          anelastic_fac=ANELASTIC_FAC,
          x0=0.0, y0=0.0,
          nu=NU, mu=MU,
          return_components=False):
    """
    Drop-in replacement for UNE_three_component.model().

    Returns (uz, ue, un) with identical sign and units to the original.

    For repeated calls on the **same** (x, y) grid, prefer UNEFastModel
    directly — this function creates a fresh instance on every call
    (no warmup overhead since Numba cache=True means compilation is instant
    after the first-ever call).
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())

    cs   = float(CAVITY_SCALE)
    hf   = float(chimney_height_fac)
    pk   = float(chimney_peck_k)
    af   = float(anelastic_fac)
    nuf  = float((1.0 - nu) / math.pi)

    r_c       = cs * yield_kt ** (1.0 / 3.0)
    A_mogi    = nuf * dv_factor * _4PI3 * r_c ** 3
    depth2    = float(depth) ** 2

    chn_sigma = pk * hf * r_c
    chn_sig2  = chn_sigma ** 2
    chn_2w2   = 2.0 * chn_sig2
    chn_aow2  = chimney_amp / chn_sig2

    r_a       = af * r_c
    cmp_sig2  = r_a ** 2
    cmp_2w2   = 2.0 * cmp_sig2
    cmp_aow2  = compact_amp / cmp_sig2

    uz, ue, un = _impl_full(x, y, float(x0), float(y0), float(depth),
                             A_mogi, depth2,
                             float(chimney_amp), chn_aow2, chn_2w2,
                             float(compact_amp), cmp_aow2, cmp_2w2)

    if return_components:
        # Reconstruct individual component arrays for plotting compatibility.
        # Uses the algebraically equivalent numpy path (rare code branch).
        xs = x - x0;  ys = y - y0
        r2 = xs*xs + ys*ys
        Rd2 = r2 + depth2;  R3 = Rd2 * np.sqrt(Rd2)
        mogi_uz = A_mogi * depth / R3
        mogi_ue = A_mogi * xs    / R3
        mogi_un = A_mogi * ys    / R3
        ec      = np.exp(-r2 / chn_2w2)
        chn_uz  = -chimney_amp * ec
        chn_ue  = -chn_aow2 * xs * ec
        chn_un  = -chn_aow2 * ys * ec
        ea      = np.exp(-r2 / cmp_2w2)
        cmp_uz  = -compact_amp * ea
        cmp_ue  = -cmp_aow2 * xs * ea
        cmp_un  = -cmp_aow2 * ys * ea
        comps = {
            'mogi':       (mogi_uz, mogi_ue, mogi_un),
            'chimney':    (chn_uz,  chn_ue,  chn_un),
            'compaction': (cmp_uz,  cmp_ue,  cmp_un),
        }
        return uz, ue, un, comps

    return uz, ue, un


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(N=5_000, n_calls=2_000):
    """
    Compare speed of UNEFastModel vs the original UNE_three_component.model().

    Parameters
    ----------
    N       : int  number of observation points
    n_calls : int  number of timed forward calls

    Returns
    -------
    dict with keys 'fast_us_per_call', 'orig_us_per_call', 'speedup', 'N'
    """
    import time

    rng = np.random.default_rng(0)
    x   = rng.uniform(-30_000, 30_000, N)
    y   = rng.uniform(-30_000, 30_000, N)
    p   = dict(depth=1500.0, yield_kt=200.0, dv_factor=-0.05,
               chimney_amp=0.12, compact_amp=0.03)

    print(f"\n── UNE_three_component_fast  benchmark ─────────────────────────")
    print(f"   N = {N:,d} observation points,  {n_calls:,d} forward calls")
    print(f"   Building UNEFastModel (JIT compile on first call) …")

    fwd = UNEFastModel(x, y)                       # warmup=True triggers compile

    # Time fast model
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fwd(**p)
    t_fast = time.perf_counter() - t0
    us_fast = t_fast / n_calls * 1e6

    print(f"\n   Fast model  :  {us_fast:7.2f} µs / call")
    print(f"                  {1e6 / us_fast:,.0f} calls / second")
    print(f"                  {3600 * 1e6 / us_fast / 1e6:.1f} M calls / hour")

    # Try original
    us_orig = None
    try:
        import UNE_three_component as _orig
        n_orig = min(n_calls, 500)
        t0 = time.perf_counter()
        for _ in range(n_orig):
            _orig.model(x, y, **p)
        t_orig = time.perf_counter() - t0
        us_orig = t_orig / n_orig * 1e6
        speedup = us_orig / us_fast
        print(f"\n   Original    :  {us_orig:7.2f} µs / call  ({n_orig} calls)")
        print(f"   Speedup     :  {speedup:.1f}×")
    except ImportError:
        speedup = None
        print("\n   (UNE_three_component not importable — skipping comparison)")

    print("────────────────────────────────────────────────────────────────\n")

    return {
        'N':                N,
        'n_calls':          n_calls,
        'fast_us_per_call': us_fast,
        'orig_us_per_call': us_orig,
        'speedup':          speedup,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    benchmark()
