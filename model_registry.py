"""
Central MODEL_REGISTRY — the single place to register a new forward model.

To add a model, add ONE entry to MODEL_REGISTRY at the bottom of this file:

    add forward model with signature: callable(X, Y, params_dict) -> (ue, un, uv)
    add prior function with signature: callable(params_dict) -> 0.0 or -np.inf
    add canonical parameter name list (for plotting)

    'my_model': {
        'forward':        callable(X, Y, params_dict) -> (ue, un, uv),
        'prior':          callable(params_dict) -> 0.0 or -np.inf,
        'param_names':    ['X0', 'Y0', ...],          # canonical plot ordering
        'default_params': {'X0': 0, 'Y0': 0, ...},   # for gen_synthetic_data
    }

All other code (MCMC, SA, plotting) imports MODEL_REGISTRY from here.
"""

import numpy as np
import pCDM_model as pCDM_fast
import okada_model as okada
import UNE_three_component_fast as UNE_three
import SBI_testing.mogi_mctigue_model as mogi_mctigue
import UNE_three_component_new_edits as UNE_three_new
import UNE_three_component_new_depth_scaling as UNE_three_new_depth_scaling
import UNE_three_component_knothe as UNE_knothe
import UNE_forward_CDM_collapse as UNE_cdm_collapse
import UNE_forward_CDM_stack as UNE_cdm_stack
import UNE_forward_CDM_stack_mogi as UNE_mogi_stack
import UNE_forward_CDM_stack_mogi_edited as UNE_two_point


# Number of closing cracks in the collapse column.  A MODEL-STRUCTURE choice,
# not a continuous parameter: an integer cannot be explored by a Gaussian
# random walk, so it is fixed here (and set explicitly in the run script).
# The cracks are EQUALLY SPACED, one at the centre of each of n equal depth
# intervals, so the stack approximates continuous closure along the column
# with an error falling as n^-2.  Against a 1024-crack reference, n = 16 is
# within 0.02 mm in LOS for a typical geometry (d = 400 m, r_c = 21 m, 41 mm
# peak) and within 0.2 mm at the most demanding contained geometry
# (d / r_c = 8.6, 120 mm peak); n = 8 would be 0.07 and 0.8 mm at half the
# cost.
UNE_CDM_STACK_N_COLLAPSE = 25


# How the closing volume is distributed along the chimney: w(zeta) ~ zeta**p,
# zeta running 0 at the cavity roof to 1 at the chimney tip.  Set to 0: the
# volume closes EQUALLY across the chimney, each crack closing dV_collapse / n.
# A positive value would weight closure toward the tip, but p is not separable
# from `depth` (both move the closure centroid), so the simplest assumption is
# used.
UNE_CDM_STACK_LOSS = 0.0


# Chimney height in cavity radii, H_c = eta * r_c.  FIXED at a measured value
# rather than sampled, and rather than derived from a sampled bulking porosity,
# because it is not identifiable: eta enters the forward model only through
# H_c, which moves the TOP of the collapse column, while `depth` moves the
# whole column -- and the two trade almost exactly.  Sweeping the equivalent
# porosity over 0.15-0.40 and refitting `depth` alone leaves 0.01-0.06 mm rms
# on a 7.6 mm signal.  Whichever of the pair is sampled therefore returns its
# prior, and with phi_bulk sampled the induced prior on eta has median 4.44
# with 8.5 per cent of its mass already inside the 4.8-5.5 literature band --
# it manufactures the agreement one would want to test.
#
# 5.5 is the top of the measured range at the NNSS (4.8-5.5 cavity radii;
# Olsen 1993, Carle et al. 2021), 4-6 in hard rock (Boardman et al. 1964).
# The bulking porosity it implies is reported instead, by
# UNE_forward_CDM_stack.report_chimney_geometry.  Read that number with the
# caveat in `phi_bulk_from_budget`: it is built from dV_collapse (well
# determined) and V_cav, hence r_c CUBED (poorly determined), so it is not an
# independent measurement of porosity unless r_c is pinned from outside.
UNE_CDM_STACK_HEIGHT_FAC = 5.5

# Largest compaction the caved rubble column can supply.  Not a free choice:
# the pore space in the column IS the cavity void, redistributed through the
# column by bulking, so phi * V_chimney = V_cav and
#
#     max_compaction = V_cav / V_chimney = 4 / (3 beta^2 eta) = 0.168
#
# for beta = 1.2, eta = 5.5.  The cap dV_collapse < max_compaction * V_chimney
# is therefore exactly dV_collapse < V_cav: the column cannot close more pore
# space than the cavity gave it.  It sets the floor on r_c,
# r_c > (dV_collapse / (max_compaction pi beta^2 eta))^(1/3) = 13.1 m for a
# fitted dV_collapse of 9343 m^3.
UNE_CDM_STACK_MAX_COMPACTION = 4.0 / (
    3.0 * UNE_cdm_stack.CHIMNEY_RADIUS_FAC ** 2 * UNE_CDM_STACK_HEIGHT_FAC)

# Bulking porosity of the caved rubble, used ONLY by 'une_mogi_stack_budget',
# where it is fixed at a literature value instead of derived.  With eta also
# fixed, the budget closes: dV_collapse = f * V_cav with
# f = 1 - 0.75 phi_b eta, so the collapse volume is no longer free and r_c is
# determined by the collapse amplitude.  Note f -> 0 at phi_b = 4/(3 eta) =
# 0.2424 (the arrest porosity) and r_c scales as f^(-1/3), so values close to
# that give an unusable r_c: phi_b = 0.15/0.20/0.24 imply r_c = 18/23/61 m for
# the same signal.  Keep it well below 0.242.
UNE_CDM_STACK_PHI_BULK = 0.20

# ── Forward wrappers ────────────────────────────────────────────────────────
def _forward_pcdm(X, Y, p):
    return pCDM_fast.pCDM(X, Y,
                          float(p['X0']), float(p['Y0']), float(p['depth']),
                          float(p['omegaX']), float(p['omegaY']), float(p['omegaZ']),
                          float(p['DVx']), float(p['DVy']), float(p['DVz']), 0.25)

def _forward_okada(X, Y, p):
    return okada.disloc3d3(X, Y,
                           xoff=float(p['X0']), yoff=float(p['Y0']),
                           depth=float(p['depth']),
                           length=float(p['length']), width=float(p['width']),
                           slip=float(p['slip']), opening=float(p['opening']),
                           strike=float(p['strike']), dip=float(p['dip']),
                           rake=float(p['rake']), nu=0.25)

def _forward_une(X, Y, p):
    uv, ue, un = UNE_three.model(
        X, Y, depth=float(p['depth']), yield_kt=float(p['yield_kt']),
        dv_factor=float(p['dv_factor']), chimney_amp=float(p['chimney_amp']),
        chimney_height_fac=10, chimney_peck_k=0.35,
        compact_amp=float(p['compact_amp']), anelastic_fac=5,
        x0=float(p['X0']), y0=float(p['Y0']), nu=0.25, mu=30e9)
    return ue, un, uv

def _forward_mogi(X, Y, p):
    return mogi_mctigue.mogi(X, Y,
                             X0=float(p['X0']), Y0=float(p['Y0']),
                             depth=float(p['depth']),
                             DV=float(p['DV']), nu=0.25)

def _forward_mctigue(X, Y, p):
    return mogi_mctigue.mctigue(X, Y,
                                X0=float(p['X0']), Y0=float(p['Y0']),
                                depth=float(p['depth']),
                                DV=float(p['DV']),
                                a=float(p['a']), c=float(p['c']),
                                nu=0.25)

def _forward_une_new(X, Y , p):
    uz, ue, un = UNE_three_new.model(X, Y, depth=p['depth'], yield_kt=p['yield_kt'],
                       dv_factor=p['dv_factor'],
                       chimney_amp=p['chimney_amp'],
                       chimney_height_fac=5.5,
                       chimney_peck_k=0.35,
                       chimney_x=p['chimney_x'],
                       chimney_y=p['chimney_y'],
                       chimney_sigma_ratio=p['chimney_sigma_ratio'],
                       chimney_rotation_deg=p['chimney_rotation_deg'],
                       compact_amp=p['compact_amp'],
                       anelastic_fac=0,
                       use_anelastic=False,
                       x0=p['X0'], y0=p['Y0'],
                       nu=0.25 , mu=30e9)

    return ue,un, uz

def _forward_une_new_depth_scaling(X, Y, p):
    uz, ue, un = UNE_three_new_depth_scaling.model(X, Y,
          depth=p['depth'],
          cavity_radius_m=p['cavity_radius_m'],
          dv_factor=p['dv_factor'],
          chimney_volume_fraction=p['chimney_volume_fraction'],
          chimney_height_fac=5.5,
          chimney_peck_k=0.35,
          chimney_x=p['chimney_x'],
          chimney_y=p['chimney_y'],
          chimney_sigma_ratio=p['chimney_sigma_ratio'],
          chimney_rotation_deg=p['chimney_rotation_deg'],
          x0=p['X0'], y0=p['Y0'],
          nu=0.25,
          return_components=False)

    return ue, un, uz


def _forward_une_cdm_collapse(X, Y, p):
    """Two-stage UNE model (UNE_forward_CDM_collapse.py): stage 1 is a McTigue
    finite-sphere co-explosive uplift source; stage 2 is a distributed elastic
    closing column ('column' collapse_model -- the physical option, per that
    module's own docstring; the alternative 'gaussian' shape-fit is not exposed
    here). height_fac/loss_exponent/gamma_deg are left at the module's own
    defaults -- height_fac and phi_bulk are NOT independent (see
    UNE_forward_CDM_collapse.CollapseBudget), so only phi_bulk is a free param.
    chimney_anchor is left at the module default ('cavity': the column is
    rooted on the cavity roof at (X0,Y0) and its axis migrates to
    (chimney_x, chimney_y) at the tip) -- 'rigid' is a categorical choice, not
    something to explore continuously via MCMC. drift_exponent (how the axis
    deviates with height) IS exposed as a free parameter.

    The column is held CIRCULAR (axis_ratio=1, rotation_deg=0). Those two were
    free parameters initially and neither was identifiable from a single-LOS
    interferogram: rotation_deg came back at 0.8 +/- 52.0 deg, i.e. spanning
    essentially its whole prior, because rotating a circle is a no-op and
    axis_ratio's own posterior included near-circular values -- so the
    rotation marginal averaged over a large region where it has no effect on
    the forward model at all. Resolving both an ellipticity and its azimuth
    from one line-of-sight projection needs a second viewing geometry.
    """
    ue, un, uz = UNE_cdm_collapse.model(
        X, Y,
        depth=float(p['depth']), r_c=float(p['r_c']), nu=0.25,
        dv_factor=float(p['dv_factor']), source_aspect=1.0,
        x0=float(p['X0']), y0=float(p['Y0']),
        collapse_model='column',
        phi_bulk=float(p['phi_bulk']),
        chimney_x=float(p['chimney_x']), chimney_y=float(p['chimney_y']),
        axis_ratio=1.0, rotation_deg=0.0,
        drift_exponent=float(p['drift_exponent']),
    )
    return ue, un, uz


def _forward_une_cdm_stack(X, Y, p):
    """Stacked-CDM UNE model (UNE_forward_CDM_stack.py).  Two stages, every
    source a compound dislocation model (Nikkhoo et al. 2017): layer 0 is the
    co-explosive uplift CDM at the working point, and layers 1..N are closing
    cracks (az = 0 CDMs) stacked up the chimney column between the cavity roof
    and the chimney tip.

    Fixed here rather than sampled:
      n_collapse    -- integer model structure, see UNE_CDM_STACK_N_COLLAPSE.
      loss_exponent -- UNE_CDM_STACK_LOSS, fixed at 1: it trades against
                       `depth` along a ridge flat to ~0.2 mm, so only their
                       combination (the closure centroid depth) is
                       constrained; `depth` carries it.
      height_fac    -- not independent of phi_bulk (the bulking budget), so
                       phi_bulk is the free parameter instead, exactly as in
                       _forward_une_cdm_collapse.
    The column is circular in plan and its axis straight and uniformly
    tilted.  Ellipticity, azimuth and axis curvature were removed rather than
    fixed: none is identifiable from a single viewing geometry, and each cost
    the sampler a badly behaved marginal.

    dV_uplift and dV_collapse are the volumes opened and closed AT THE SOURCE,
    in m^3. For the closing cracks of the column the volume of the resulting
    surface bowl equals the volume closed; for the equant uplift source it is
    (2/3)(1+nu) of it. No conversion is applied anywhere.
    """
    # Strength comes in as VOLUMES when they are given (the 'une_cdm_stack_vol'
    # parameterisation), otherwise as fractions of V_cav (the 'une_cdm_stack'
    # one, matching une_cdm_collapse). dV_uplift/dV_collapse are preferred for
    # inversion: dv_factor is degenerate with r_c to within the finite-source
    # correction, since layer 0's field depends only on dv_factor * V_cav(r_c).
    r_c = float(p['r_c'])
    uplift_dv = float(p['dV_uplift']) if 'dV_uplift' in p else None
    collapse_dv = float(p['dV_collapse']) if 'dV_collapse' in p else None
    ue, un, uz = UNE_cdm_stack.model(
        X, Y, nu=0.25,
        depth=float(p['depth']), r_c=r_c,
        dv_factor=float(p.get('dv_factor', UNE_cdm_stack.DV_FACTOR)),
        uplift_dv=uplift_dv, collapse_dv=collapse_dv,
        # The co-explosive source is a full CDM: two shape ratios and two
        # tilts, with r_c setting its size as the equal-volume sphere radius.
        # Equant (both ratios 1) would make it isotropic and so
        # indistinguishable from a Mogi point source, which is what fixing the
        # shape had quietly assumed.  All three CDM rotations are carried,
        # but expect very different resolution: at this depth the azimuth
        # omegaZ moves the field by 0.01 mm for an axisymmetric source and
        # 0.56 mm for a triaxial one, against 6-10 mm for either tilt, so it
        # is there for completeness rather than because the data can pin it.
        aspect_y=float(p.get('aspect_y', 1.0)),
        aspect_z=float(p.get('aspect_z', 1.0)),
        source_omegaX=float(p.get('source_omegaX', 0.0)),
        source_omegaY=float(p.get('source_omegaY', 0.0)),
        source_omegaZ=float(p.get('source_omegaZ', 0.0)),
        x0=float(p['X0']), y0=float(p['Y0']),
        n_collapse=UNE_CDM_STACK_N_COLLAPSE,
        # The bulking porosity is an INPUT and the chimney height a derived
        # product: phi_b = (1 - dV_collapse/V_cav) / (0.75 eta).  This is the
        # way round that does not sample an unidentifiable direction: eta acts
        # only through H_c, which trades against `depth`, so sampling the pair
        # returns the prior.  See UNE_CDM_STACK_HEIGHT_FAC.
        height_fac=UNE_CDM_STACK_HEIGHT_FAC, derive_height=False,
        chimney_x=float(p['chimney_x']), chimney_y=float(p['chimney_y']),
        # where the closure sits within the column: w(zeta) ~ zeta**p, zeta
        # running 0 at the cavity roof to 1 at the chimney tip.  0 is uniform
        # closure, larger values concentrate it toward the tip, where void
        # accumulates once caving stalls.  NOT independently identifiable:
        # with both stage volumes free it trades against `depth` along a
        # ridge flat to ~0.2 mm, so report the closure centroid depth
        # (UNE_forward_CDM_stack.closure_centroid_depth) rather than p.
        loss_exponent=float(p.get('loss_exponent',
                                  UNE_CDM_STACK_LOSS)),
    )
    return ue, un, uz


def _forward_une_cdm_stack_collapse(X, Y, p):
    """Stacked-CDM UNE model with the CO-EXPLOSIVE UPLIFT STAGE REMOVED.

    Same forward code as _forward_une_cdm_stack (UNE_forward_CDM_stack.py) and
    the same collapse column, built by the same build_stack call -- only layer
    0, the uplift CDM at the working point, is switched off
    (include_uplift=False, uplift_dv=0).  The source is therefore purely the
    quadrature of closing cracks up the chimney.

    This is the model for a POST-EXPLOSIVE interferogram, i.e. one whose first
    acquisition is after the shot, so the co-explosive uplift has already
    happened and is not in the data.  It is exactly the situation
    UNE_forward_CDM_stack's own docstring describes for JUNCTION: Foxall
    (2000, UCRL-JC-138986) fitted that subsidence bowl with stacked closing
    sources and no uplift stage, because the interferogram starts a month
    after the event.  Divider needs the uplift layer; JUNCTION does not.

    Five parameters of 'une_cdm_stack_vol' are therefore dropped rather than
    fixed -- dV_uplift and the shape/orientation of the uplift CDM
    (aspect_y, aspect_z, source_omegaX/Y/Z) -- since with no uplift layer
    none of them touches the forward model at all; leaving them in the
    sampler would return the prior.  The nine that remain (X0, Y0, depth,
    r_c, dV_collapse, phi_bulk, chimney_x, chimney_y and the fixed
    structural constants) mean exactly what they mean in the two-stage runs,
    so the collapse stage is directly comparable between them.

    Note what `depth` still is: the WORKING POINT depth.  The column hangs
    below it, from the cavity roof (depth - r_c) up to the chimney tip
    (depth - eta r_c), so depth remains meaningful and identifiable through
    the column's position even though no source sits at it.
    """
    return UNE_cdm_stack.model(
        X, Y, nu=0.25,
        depth=float(p['depth']), r_c=float(p['r_c']),
        # layer 0 off: both switches, so the intent survives either being
        # read on its own.
        include_uplift=False, uplift_dv=0.0,
        collapse_dv=float(p['dV_collapse']),
        x0=float(p['X0']), y0=float(p['Y0']),
        n_collapse=UNE_CDM_STACK_N_COLLAPSE,
        # Identical to the two-stage wrapper: the chimney height is the
        # INPUT (fixed at a measured value) and the bulking porosity
        # phi_b = (1 - dV_collapse/V_cav) / (0.75 eta) the derived product.
        height_fac=UNE_CDM_STACK_HEIGHT_FAC, derive_height=False,
        chimney_x=float(p['chimney_x']), chimney_y=float(p['chimney_y']),
        loss_exponent=float(p.get('loss_exponent', UNE_CDM_STACK_LOSS)),
    )


def _forward_une_mogi_stack(X, Y, p):
    """Reduced form of _forward_une_cdm_stack: the co-explosive source is an
    ISOTROPIC Mogi point source instead of a full compound dislocation model.
    The collapse column is unchanged -- the same quadrature of thickness-free
    closing CDMs, built by the same call -- so the two models differ in the
    uplift stage and in nothing else, and a comparison between the two runs
    isolates that stage.

    Five parameters are dropped: aspect_y, aspect_z and the three rotations.
    They are dropped rather than fixed because they are not identifiable.
    Driving the full CDM to a strongly triaxial shape and then refitting only
    dV_uplift and depth with an equant source leaves a residual of 0.05 to
    0.30 mm rms (max 1.5 mm) across shapes from 5:1 prolate to 1000:1
    flattened, at the divider working point where the uplift stage itself is
    only ~8 mm peak-to-peak. The shape parameters do not measure shape; they
    re-parameterise volume and depth, and the refit shows the exchange rate:
    a source flattened to az/ax = 0.2 is read as 1.77x the volume, 133 m
    deeper. The azimuth is weaker again -- 0.03 mm at 30 deg, against 0.36
    and 0.30 mm for the two tilts -- because at 24 cavity radii' depth the
    source is a point and a point has no azimuth.

    Run UNE_forward_CDM_stack_mogi.aspect_sensitivity() at the posterior mode
    of the full-CDM inversion to regenerate those numbers for the parameters
    the data actually chose, rather than for an assumed working point.

    The co-explosive strength is sampled as dV_eff, Mogi's effective volume
    change -- a strength, not the volume of the cavity -- so the uplift is the
    textbook (1 - nu) dV_eff / (pi R^3) (dx, dy, d).  It is converted
    internally to the potency mogi_point takes (1.8 dV_eff at nu = 0.25); to
    compare with the CDM run's dV_uplift, multiply dV_eff by 1.8.

    Note one intended behavioural difference: r_c no longer touches the
    uplift field, because a point source has no size. Here it enters only
    through the chimney -- the cavity volume in the bulking budget, the
    column radius 1.2 r_c, and the column's depth extent. That removes the
    0.3 per cent finite-source correction, which was the weakest of the
    handles on r_c; the collapse stage carries it.
    """
    ue, un, uz = UNE_mogi_stack.model(
        X, Y, nu=0.25,
        depth=float(p['depth']), r_c=float(p['r_c']),
        dv_factor=float(p.get('dv_factor', UNE_mogi_stack.DV_FACTOR)),
        uplift_dv=(float(p['dV_uplift']) if 'dV_uplift' in p else None),
        uplift_dv_eff=(float(p['dV_eff']) if 'dV_eff' in p else None),
        collapse_dv=(float(p['dV_collapse']) if 'dV_collapse' in p else None),
        x0=float(p['X0']), y0=float(p['Y0']),
        n_collapse=UNE_CDM_STACK_N_COLLAPSE,
        # chimney height in, bulking porosity out -- exactly as in the CDM
        # version, so the derived phi_b is comparable between the two runs.
        height_fac=UNE_CDM_STACK_HEIGHT_FAC, derive_height=False,
        chimney_x=float(p['chimney_x']), chimney_y=float(p['chimney_y']),
        loss_exponent=float(p.get('loss_exponent', UNE_CDM_STACK_LOSS)),
    )
    return ue, un, uz


def _forward_une_knothe(X, Y, p):
    uz, ue, un = UNE_knothe.model(X, Y,
          depth=p['depth'],
          cavity_radius_m=p['cavity_radius_m'],
          dv_factor=p['dv_factor'],
          chimney_volume_fraction=p['chimney_volume_fraction'],
          chimney_height_fac=5.5,
          knothe_influence_angle_deg=p['knothe_influence_angle_deg'],
          knothe_elastic_floor=0.2,
          chimney_x=p['chimney_x'],
          chimney_y=p['chimney_y'],
          chimney_sigma_ratio=p['chimney_sigma_ratio'],
          chimney_rotation_deg=p['chimney_rotation_deg'],
          x0=p['X0'], y0=p['Y0'],
          nu=0.25,
          return_components=False)
    return ue, un, uz



# ── Prior constraints ───────────────────────────────────────────────────────
def _prior_pcdm(p):
    return 0.0

def _prior_okada(p):
    if 'length' in p and p['length'] <= 0:
        return -np.inf
    if 'width' in p and p['width'] <= 0:
        return -np.inf
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    # depth is the fault CENTROID depth (okada_model.py: "fault centroid located at
    # (0,0,-DEPTH)"), not the top-edge depth -- a positive centroid depth alone does not
    # stop the top edge from breaking the surface for a shallow/wide/steep fault (e.g.
    # depth=100, width=10000, dip=60 gives a top edge at -4230 m). Use abs(sin(dip)) so
    # this is robust regardless of dip's sign convention (dip can be sampled negative to
    # match GBIS's native fault-model convention elsewhere in this codebase).
    if 'depth' in p and 'width' in p and 'dip' in p:
        top_depth = p['depth'] - (p['width'] / 2) * abs(np.sin(np.radians(p['dip'])))
        if top_depth <= 0:
            return -np.inf
    return 0.0

def _prior_une(p):
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'yield_kt' in p and p['yield_kt'] <= 0:
        return -np.inf
    return 0.0

def _prior_une_new(p):
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'yield_kt' in p and p['yield_kt'] <= 0:
        return -np.inf
    return 0.0

def _prior_mogi_yang(p):
    if 'DV' in p and abs(p['DV']) > 1e10:
        return -np.inf
    return 0.0

def _prior_mctigue(p):
    if 'a' in p and 'c' in p:
        if p['a'] <= 0 or p['c'] <= 0:
            return -np.inf
        if p['a'] > 10000 or p['c'] > 10000:
            return -np.inf
    return 0.0

def _prior_une_new_depth_scaling(p):
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'cavity_radius_m' in p and p['cavity_radius_m'] <= 0:
        return -np.inf
    return 0.0

def _prior_une_cdm_collapse(p):
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'r_c' in p and p['r_c'] <= 0:
        return -np.inf
    if 'dv_factor' in p and p['dv_factor'] <= 0:
        return -np.inf
    # bulking porosity of caved rubble -- module's own warn() flags 0.10-0.40 as the
    # physically expected range; allow a bit either side rather than hard-clamping there
    if 'phi_bulk' in p and not (0.05 <= p['phi_bulk'] <= 0.5):
        return -np.inf
    # The residual void fraction must be POSITIVE. f_resid = 1 - 0.75*phi_bulk*height_fac
    # crosses zero at phi_bulk = 4/(3*height_fac) (= 0.2424 for the height_fac = 5.5 the
    # forward wrapper uses). Above that, CollapseBudget returns V_resid < 0 and the
    # closing column silently flips sign -- the "collapse" becomes an inflation source.
    # CollapseBudget.warn() flags this but is never called on the forward path, so the
    # sampler is otherwise free to wander into that unphysical region (it did: a
    # phi_bulk prior of (0.1, 0.4) put ~60% of its range there).
    if 'phi_bulk' in p:
        f_resid = 1.0 - 0.75 * p['phi_bulk'] * UNE_cdm_collapse.CHIMNEY_HEIGHT_FAC
        if f_resid <= 0:
            return -np.inf
    if 'drift_exponent' in p and p['drift_exponent'] <= 0:
        return -np.inf
    return 0.0


def _prior_une_cdm_stack(p):
    """Same physical bounds as _prior_une_cdm_collapse -- the parameters mean
    the same things, only the elastic source type differs."""
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'r_c' in p and p['r_c'] <= 0:
        return -np.inf
    if 'dv_factor' in p and p['dv_factor'] <= 0:
        return -np.inf
    if 'phi_bulk' in p and not (0.05 <= p['phi_bulk'] <= 0.5):
        return -np.inf
    # residual void fraction must stay positive, else the closing layers flip
    # sign and the "collapse" becomes an inflation source (collapse_volume()
    # clips at zero, which would instead delete the stage entirely)
    if 'phi_bulk' in p:
        f_resid = 1.0 - 0.75 * p['phi_bulk'] * UNE_cdm_stack.CHIMNEY_HEIGHT_FAC
        if f_resid <= 0:
            return -np.inf
    if 'loss_exponent' in p and not (0.0 <= p['loss_exponent'] <= 10.0):
        return -np.inf
    return 0.0


def _prior_une_cdm_stack_vol(p):
    """Volume parameterisation of the stacked-CDM model.

    Strength is dV_uplift / dV_collapse in m^3, with phi_bulk supplied and the
    chimney height derived from the budget: the residual void that closes cannot exceed the cavity
    void it came from, and the implied bulking porosity

        phi_bulk = (1 - dV_collapse / V_cav) / (0.75 * height_fac)

    must stay above ~0.05 (caved rubble always bulks). With height_fac = 5.5
    that caps dV_collapse at 0.794 * V_cav. dV_uplift is deliberately NOT
    bounded by V_cav: the co-explosive and collapse stages are independent
    sources that share only X0, Y0, depth and r_c.
    """
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'r_c' in p and p['r_c'] <= 0:
        return -np.inf
    if 'dV_uplift' in p and p['dV_uplift'] <= 0:
        return -np.inf
    if 'dV_collapse' in p and p['dV_collapse'] <= 0:
        return -np.inf
    # Shape of the co-explosive CDM.  The bounds keep the source within a
    # factor of five of equant in either direction; beyond that it is a sheet
    # or a needle rather than a cavity, and the small-strain description of a
    # dislocation source with an opening comparable to its own short axis
    # starts to fail.
    # Shape of the co-explosive CDM.  These bounds are deliberately wide, so
    # that the box priors in the run script decide the range rather than this
    # guard; they exist only to keep the source from degenerating entirely.
    # Note that a very flat source approaches the small-strain limit of a
    # dislocation description, its opening becoming comparable to its own
    # short semi-axis: at aspect_z = 0.01 and r_c = 23 m the source is 2 m
    # thick and an uplift volume of 5e4 m^3 would open it by about its own
    # half-thickness.
    for key in ('aspect_y', 'aspect_z'):
        if key in p and not (0.005 <= p[key] <= 20.0):
            return -np.inf
    for key in ('source_omegaX', 'source_omegaY', 'source_omegaZ'):
        if key in p and not (-180.0 <= p[key] <= 180.0):
            return -np.inf
    if 'r_c' in p:
        V_cav = UNE_cdm_stack.cavity_volume(p['r_c'])
        # What settles under gravity is the PORE SPACE of the caved rubble,
        # so the closing volume is capped by the compaction the column can
        # supply: dV_collapse <= max_compaction * V_chimney, with
        # V_chimney = pi beta^2 eta r_c^3.  The chimney must also stop below
        # the free surface.
        if 'dV_collapse' in p:
            V_chim = (np.pi * UNE_cdm_stack.CHIMNEY_RADIUS_FAC ** 2
                      * UNE_CDM_STACK_HEIGHT_FAC * p['r_c'] ** 3)
            if p['dV_collapse'] >= UNE_CDM_STACK_MAX_COMPACTION * V_chim:
                return -np.inf
        if 'depth' in p and UNE_CDM_STACK_HEIGHT_FAC * p['r_c'] >= p['depth']:
            return -np.inf
    if 'loss_exponent' in p and not (0.0 <= p['loss_exponent'] <= 10.0):
        return -np.inf
    return 0.0


def _prior_une_mogi_stack_vol(p):
    """_prior_une_cdm_stack_vol with the shape and rotation guards removed.

    Every physical bound is identical -- same volume budget, same bulking
    porosity range, same admissibility condition on the derived chimney
    height -- so the two models are compared under the same prior on the
    nine parameters they share, and the comparison measures the extra five
    rather than a change of prior.
    """
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'dV_eff' in p and p['dV_eff'] <= 0:
        return -np.inf
    if 'r_c' in p and p['r_c'] <= 0:
        return -np.inf
    if 'dV_uplift' in p and p['dV_uplift'] <= 0:
        return -np.inf
    if 'dV_collapse' in p and p['dV_collapse'] <= 0:
        return -np.inf
    if 'r_c' in p:
        V_cav = UNE_cdm_stack.cavity_volume(p['r_c'])
        # What settles under gravity is the PORE SPACE of the caved rubble,
        # so the closing volume is capped by the compaction the column can
        # supply: dV_collapse <= max_compaction * V_chimney, with
        # V_chimney = pi beta^2 eta r_c^3.  The chimney must also stop below
        # the free surface.
        if 'dV_collapse' in p:
            V_chim = (np.pi * UNE_cdm_stack.CHIMNEY_RADIUS_FAC ** 2
                      * UNE_CDM_STACK_HEIGHT_FAC * p['r_c'] ** 3)
            if p['dV_collapse'] >= UNE_CDM_STACK_MAX_COMPACTION * V_chim:
                return -np.inf
        if 'depth' in p and UNE_CDM_STACK_HEIGHT_FAC * p['r_c'] >= p['depth']:
            return -np.inf
    if 'loss_exponent' in p and not (0.0 <= p['loss_exponent'] <= 10.0):
        return -np.inf
    return 0.0



def _forward_une_two_point(X, Y, p):
    """Two-point-source UNE model (UNE_forward_CDM_stack_mogi_edited.py).

    An isotropic Mogi point source for the co-explosive stage at
    (X0, Y0, depth), and a point tensile crack -- the zero-size limit of the
    horizontal closing crack used in UNE_forward_CDM_stack -- for the collapse
    stage at (chimney_x, chimney_y, chimney_depth).  The two depths are
    sampled independently: neither the cavity radius r_c nor any chimney
    geometry (eta, beta, p, n) enters the forward model at all.

    Why: in the chimney-column model, `depth` and `r_c` jointly place the
    column and trade off almost exactly (sweeping r_c 12-45 m and refitting
    depth alone reproduces the field to 0.01-0.06 mm rms), so r_c returns its
    prior.  A single point crack at the column's own centroid depth
    reproduces the full column's LOS field to 0.002-0.09 mm rms for
    r_c = 15-60 m, so nothing measurable is lost by collapsing it to a point,
    and the depth the data actually constrain -- the collapse centroid --
    becomes a directly sampled parameter instead of a combination of two
    confounded ones.

    dV_eff is Mogi's effective volume change of the co-explosive source (a
    strength, not a cavity volume); dV_collapse is the volume closed by the
    point crack.  Both are positive (the closing sign is applied inside the
    model).  For the point
    crack the bowl factor is exactly 1, so dV_collapse is directly the
    volume of the subsidence bowl.

    r_c is NOT sampled.  If wanted, report it after the run as
    UNE_forward_CDM_stack_mogi_edited.implied_r_c(depth, chimney_depth),
    which is only defined where depth > chimney_depth -- see that
    function and report_two_point_geometry for why that ordering is not
    enforced here.
    """
    ue, un, uz = UNE_two_point.model(
        X, Y, nu=0.25,
        X0=float(p['X0']), Y0=float(p['Y0']),
        depth=float(p['depth']), dV_eff=float(p['dV_eff']),
        chimney_x=float(p['chimney_x']), chimney_y=float(p['chimney_y']),
        chimney_depth=float(p['chimney_depth']),
        dV_collapse=float(p['dV_collapse']))
    return ue, un, uz


def _prior_une_two_point(p):
    """Physical bounds only.

    Deliberately NO ordering constraint between `depth` (co-explosive) and
    `chimney_depth` (collapse).  A nested cavity-and-chimney picture needs
    chimney_depth < depth, but on the Divider interferogram the
    unconstrained best fit puts the collapse point ~90 m BELOW the uplift
    point, and forcing the physical order costs Delta-chi2 of 2 at equal
    depths rising to 7 at a 91 m separation.  Imposing the order here would
    hide that finding inside a prior bound; leaving it out lets the posterior
    show it, and report_two_point_geometry reports what fraction of samples
    violate it.
    """
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'chimney_depth' in p and p['chimney_depth'] <= 0:
        return -np.inf
    if 'dV_eff' in p and p['dV_eff'] < 0:
        return -np.inf
    if 'dV_collapse' in p and p['dV_collapse'] < 0:
        return -np.inf
    return 0.0


def _forward_une_mogi_stack_budget(X, Y, p):
    """'une_mogi_stack_vol' with BOTH phi_b and eta fixed at literature values.

    The budget then determines the collapse volume from the cavity alone,
    dV_collapse = (1 - 0.75 phi_b eta) V_cav(r_c), so dV_collapse is dropped as
    a free parameter and r_c is fixed by the collapse amplitude instead of by
    the weak chimney-length information.  Seven parameters, not eight.

    The price: r_c scales as f^(-1/3) with f = 1 - 0.75 phi_b eta, and f -> 0
    at phi_b = 0.2424, so the fitted r_c is a restatement of the assumed
    phi_b as much as a measurement.  See UNE_CDM_STACK_PHI_BULK.
    """
    r_c = float(p['r_c'])
    f = 1.0 - 0.75 * UNE_CDM_STACK_PHI_BULK * UNE_CDM_STACK_HEIGHT_FAC
    ue, un, uz = UNE_mogi_stack.model(
        X, Y, nu=0.25,
        depth=float(p['depth']), r_c=r_c,
        uplift_dv_eff=float(p['dV_eff']),
        collapse_dv=f * UNE_cdm_stack.cavity_volume(r_c),
        x0=float(p['X0']), y0=float(p['Y0']),
        n_collapse=UNE_CDM_STACK_N_COLLAPSE,
        height_fac=UNE_CDM_STACK_HEIGHT_FAC, derive_height=False,
        chimney_x=float(p['chimney_x']), chimney_y=float(p['chimney_y']),
        loss_exponent=UNE_CDM_STACK_LOSS)
    return ue, un, uz


def _prior_une_mogi_stack_budget(p):
    """Positivity, and the chimney must stop below the free surface."""
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'r_c' in p and p['r_c'] <= 0:
        return -np.inf
    if 'dV_eff' in p and p['dV_eff'] < 0:
        return -np.inf
    if 'r_c' in p and 'depth' in p and UNE_CDM_STACK_HEIGHT_FAC * p['r_c'] >= p['depth']:
        return -np.inf
    return 0.0


def _prior_une_knothe(p):
    if 'depth' in p and p['depth'] <= 0:
        return -np.inf
    if 'cavity_radius_m' in p and p['cavity_radius_m'] <= 0:
        return -np.inf
    if 'knothe_influence_angle_deg' in p:
        if p['knothe_influence_angle_deg'] <= 0 or p['knothe_influence_angle_deg'] >= 90:
            return -np.inf
    return 0.0


# ── Registry ────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    'pcdm': {
        'forward': _forward_pcdm,
        'prior':   _prior_pcdm,
        'param_names': ['X0', 'Y0', 'depth', 'DVx', 'DVy', 'DVz',
                        'omegaX', 'omegaY', 'omegaZ'],
        'default_params': {
            'X0': 0.5, 'Y0': -0.3, 'depth': 12000,
            'DVx': 5e7, 'DVy': 5e7, 'DVz': 5e7,
            'omegaX': 10.0, 'omegaY': -30.0, 'omegaZ': -10.0,
        },
    },
    'okada': {
        'forward': _forward_okada,
        'prior':   _prior_okada,
        'param_names': ['X0', 'Y0', 'depth', 'length', 'width',
                        'strike', 'dip', 'rake', 'slip', 'opening'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 4.0e3,
            'length': 10e3, 'width': 4e3,
            'strike': 120, 'dip': 30, 'rake': 10,
            'slip': 3, 'opening': 0.0,
        },
    },
    'une': {
        'forward': _forward_une,
        'prior':   _prior_une,
        'param_names': ['X0', 'Y0', 'depth', 'yield_kt', 'dv_factor',
                        'chimney_amp', 'compact_amp'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 700.0,
            'yield_kt': 500.0, 'dv_factor': 0.2,
            'chimney_amp': 0.25, 'compact_amp': 0.25,
        },
    },
    'mogi': {
        'forward': _forward_mogi,
        'prior':   _prior_mogi_yang,
        'param_names': ['X0', 'Y0', 'depth', 'DV'],
        'default_params': {'X0': 0, 'Y0': 0, 'depth': 5000, 'DV': 1e6},
    },
    'mctigue': {
        'forward': _forward_mctigue,
        'prior':   _prior_mctigue,
        'param_names': ['X0', 'Y0', 'depth', 'DV', 'a', 'c'],
        'default_params': {'X0': 0, 'Y0': 0, 'depth': 5000, 'DV': 1e6, 'a': 1000, 'c': 1000},
    },

    'une_ang': {
        'forward': _forward_une_new,
        'prior':   _prior_une_new,
        'param_names': ['X0', 'Y0', 'depth', 'yield_kt', 'dv_factor',
                        'chimney_amp', 
                        'chimney_x', 'chimney_y', 'chimney_sigma_ratio',
                        'chimney_rotation_deg', 'compact_amp'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 700.0,
            'yield_kt': 500.0, 'dv_factor': 0.2,
            'chimney_amp': 0.25, 'chimney_peck_k': 0.35,
            'chimney_x': 0.0, 'chimney_y': 0.0, 'chimney_sigma_ratio': 1.0,
            'chimney_rotation_deg': 0.0, 'compact_amp': 0.25,
        },
    },

    'une_depth_scaling': {
        'forward': _forward_une_new_depth_scaling,
        'prior':   _prior_une_new_depth_scaling,
        'param_names': ['X0', 'Y0', 'depth', 'cavity_radius_m', 'dv_factor',
                        'chimney_volume_fraction', 'chimney_x', 'chimney_y', 
                        'chimney_sigma_ratio', 'chimney_rotation_deg', 'chimney_arrested_expansion'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 700.0,
            'cavity_radius_m': 500.0, 'dv_factor': 0.2,
            'chimney_volume_fraction': 0.25, 
            'chimney_x': 0.0, 'chimney_y': 0.0, 'chimney_sigma_ratio': 1.0, 
            'chimney_rotation_deg': 0.0, 'chimney_arrested_expansion': 0.1,
        },
    },


    'une_knothe': {
        'forward': _forward_une_knothe,
        'prior':   _prior_une_knothe,
        'param_names': ['X0', 'Y0', 'depth', 'cavity_radius_m', 'dv_factor',
                        'chimney_volume_fraction',
                        'chimney_x', 'chimney_y',
                        'chimney_sigma_ratio', 'chimney_rotation_deg',
                        'knothe_influence_angle_deg'],
        'default_params': {
            'X0': 132.0, 'Y0': -264.0,
            'depth': 700.0, 'cavity_radius_m': 18.0,
            'dv_factor': 0.59,
            'chimney_volume_fraction': 0.071,
            'chimney_x': 132.0 + (-154.0), 'chimney_y': -264.0 + 505.0,
            'chimney_sigma_ratio': 1.73, 'chimney_rotation_deg': -21.6,
            'knothe_influence_angle_deg': 70.0,
        },
    },

    'une_cdm_collapse': {
        'forward': _forward_une_cdm_collapse,
        'prior':   _prior_une_cdm_collapse,
        'param_names': ['X0', 'Y0', 'depth', 'r_c', 'dv_factor', 'phi_bulk',
                        'chimney_x', 'chimney_y', 'drift_exponent'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 500.0, 'r_c': 20.0,
            'dv_factor': 0.05, 'phi_bulk': 0.225,
            'chimney_x': 0.0, 'chimney_y': 0.0, 'drift_exponent': 1.0,
        },
    },

    'une_cdm_stack': {
        'forward': _forward_une_cdm_stack,
        'prior':   _prior_une_cdm_stack,
        'param_names': ['X0', 'Y0', 'depth', 'r_c', 'dv_factor', 'phi_bulk',
                        'chimney_x', 'chimney_y'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 500.0, 'r_c': 20.0,
            'dv_factor': 0.05, 'phi_bulk': 0.225,
            'chimney_x': 0.0, 'chimney_y': 0.0,
        },
    },

    # Same forward model as 'une_cdm_stack', parameterised by the two stage
    # VOLUMES instead of dv_factor/phi_bulk. Prefer this one for inversion:
    # dv_factor is degenerate with r_c (only their product, the volume, is
    # observable), while r_c on its own remains meaningful geometry because
    # it sets the chimney height and radius.
    'une_cdm_stack_vol': {
        'forward': _forward_une_cdm_stack,
        'prior':   _prior_une_cdm_stack_vol,
        'param_names': ['X0', 'Y0', 'depth', 'r_c', 'dV_uplift', 'dV_collapse',
                        'aspect_y', 'aspect_z',
                        'source_omegaX', 'source_omegaY', 'source_omegaZ',
                        'chimney_x', 'chimney_y'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 500.0, 'r_c': 20.0,
            'dV_uplift': 2.0e3, 'dV_collapse': 1.5e3,
            'aspect_y': 1.0, 'aspect_z': 1.0,
            'source_omegaX': 0.0, 'source_omegaY': 0.0,
            'source_omegaZ': 0.0,
            'chimney_x': 0.0, 'chimney_y': 0.0,
        },
    },

    # Same model and the same collapse column as 'une_cdm_stack_vol', with the
    # co-explosive uplift layer switched off entirely -- the model for a
    # POST-EXPLOSIVE interferogram, which contains only the collapse (JUNCTION,
    # Foxall 2000).  dV_uplift and the four shape/orientation parameters of the
    # uplift CDM are dropped rather than fixed: with no uplift layer they do
    # not enter the forward model, so sampling them would only echo the prior.
    # The prior is _prior_une_cdm_stack_vol unchanged -- every one of its
    # dV_uplift / aspect / rotation guards is written as `if key in p`, so with
    # those keys absent it reduces exactly to the collapse-only bounds (volume
    # budget, bulking porosity, and the admissibility of the derived chimney
    # height) with nothing relaxed.
    'une_cdm_stack_vol_collapse': {
        'forward': _forward_une_cdm_stack_collapse,
        'prior':   _prior_une_cdm_stack_vol,
        'param_names': ['X0', 'Y0', 'depth', 'r_c', 'dV_collapse',
                        'chimney_x', 'chimney_y'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 622.0, 'r_c': 60.0,
            'dV_collapse': 3.0e4,
            'chimney_x': 0.0, 'chimney_y': 0.0,
        },
    },

    # Same model as 'une_cdm_stack_vol' with the co-explosive source reduced
    # to an isotropic Mogi point source: the nine parameters both models
    # share, and none of the five that describe the shape and orientation of
    # the uplift CDM.  Those five are not identifiable -- once dV_uplift and
    # depth are refitted, even a 1000:1 triaxial source is reproduced by a
    # sphere to 0.3 mm rms -- so this is the model to quote, with the
    # 14-parameter run as the evidence that nothing was lost.
    'une_mogi_stack_vol': {
        'forward': _forward_une_mogi_stack,
        'prior':   _prior_une_mogi_stack_vol,
        'param_names': ['X0', 'Y0', 'depth', 'r_c', 'dV_eff', 'dV_collapse',
                        'chimney_x', 'chimney_y'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 500.0, 'r_c': 20.0,
            'dV_eff': 1.1e3, 'dV_collapse': 1.5e3,
            'chimney_x': 0.0, 'chimney_y': 0.0,
        },
    },

    # 'une_mogi_stack_vol' with phi_b and eta both fixed at literature values,
    # so the budget closes and dV_collapse = (1 - 0.75 phi_b eta) V_cav is
    # derived from r_c rather than sampled.  r_c is then set by the collapse
    # amplitude -- but inherits the assumed phi_b, steeply (see
    # UNE_CDM_STACK_PHI_BULK).
    'une_mogi_stack_budget': {
        'forward': _forward_une_mogi_stack_budget,
        'prior':   _prior_une_mogi_stack_budget,
        'param_names': ['X0', 'Y0', 'depth', 'r_c', 'dV_eff',
                        'chimney_x', 'chimney_y'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 450.0, 'r_c': 23.0,
            'dV_eff': 5.0e3, 'chimney_x': 0.0, 'chimney_y': 0.0,
        },
    },

    # Two-point-source model: an isotropic Mogi point source (co-explosive
    # stage) and a point tensile crack (collapse stage), each at an
    # independently free depth and position.  Same parameter count as
    # 'une_mogi_stack_vol' (8), with (depth, r_c) replaced by
    # (depth, chimney_depth): no cavity radius and no chimney geometry enters
    # the forward model.  See _forward_une_two_point for the justification
    # and _prior_une_two_point for why the two depths are not ordered.
    'une_two_point': {
        'forward': _forward_une_two_point,
        'prior':   _prior_une_two_point,
        'param_names': ['X0', 'Y0', 'depth', 'dV_eff',
                        'chimney_x', 'chimney_y', 'chimney_depth',
                        'dV_collapse'],
        'default_params': {
            'X0': 0.0, 'Y0': 0.0, 'depth': 400.0, 'dV_eff': 8.0e3,
            'chimney_x': 0.0, 'chimney_y': 0.0, 'chimney_depth': 400.0,
            'dV_collapse': 9.0e3,
        },
    },

    # Placeholder — uncomment when Yang forward function is available:
    # 'yang': {
    #     'forward': _forward_yang,
    #     'prior':   _prior_mogi_yang,
    #     'param_names': ['X0', 'Y0', 'depth', 'DV', 'radius'],
    #     'default_params': {'X0': 0, 'Y0': 0, 'depth': 5000, 'DV': 1e6, 'radius': 500},
    # },
}


def get_param_names(model_type, samples=None):
    """Return canonical parameter name list for a model type.

    For multi-model (model_type is list or samples have prefixed keys),
    returns the raw sample keys.  For a single registered model, returns
    the ordered 'param_names'.  Falls back to list(samples.keys()).
    """
    if samples is not None:
        if isinstance(model_type, list) or any('__' in k for k in samples.keys()):
            return list(samples.keys())
    m = model_type.lower() if isinstance(model_type, str) else ''
    if m in MODEL_REGISTRY:
        return list(MODEL_REGISTRY[m]['param_names'])
    if samples is not None:
        return list(samples.keys())
    return []


def forward_from_registry(model_name, X, Y, params):
    """Look up *model_name* in MODEL_REGISTRY and call its forward function.

    Returns (ue, un, uv).  Raises ValueError for unknown models.
    """
    m = model_name.lower()
    if m not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Registered models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[m]['forward'](X, Y, params)
