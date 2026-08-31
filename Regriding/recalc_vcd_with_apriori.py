#!/usr/bin/env python3
"""
recalc_vcd_with_apriori.py — recompute a satellite tropospheric VCD under a
*different* a priori profile.

This is the general idea used to put two datasets (two instruments, or a model
and an instrument) on a common a priori for a fair comparison: the retrieved
slant column is left unchanged, but the air-mass-factor / averaging-kernel step
is re-applied with a NEW a priori partial-column profile.

Method  (TROPOMI NO2 Product User Manual / ATBD, Eq. 4)
-------------------------------------------------------
    VCD_trop*  =  VCD_trop  ·  Σ_trop( x_new )  /  Σ_trop( AK_trop · x_new )

with the sums over the tropospheric model layers and

    x_new    the NEW a priori partial-column profile (e.g. MUSICA, GCHP, or
             GEOS-CF), mass-conservatively interpolated in sigma = P / Ps onto
             the retrieval's own vertical grid (here TM5).
    AK_trop  the tropospheric averaging kernel on that grid. If a product stores
             the TOTAL-column AK, convert with AK_trop = AK_total · AMF_total /
             AMF_trop and zero the stratospheric layers.
    trop     layers below the tropopause pressure — use the a priori model's OWN
             tropopause, so the vertical extent is self-consistent with x_new.

Notes
-----
* The a priori enters only as a ratio, so its absolute normalization cancels —
  pass partial columns [molec cm-2] directly, no need to normalize by the total.
* The averaging kernel already encodes the cloud treatment, so no separate
  cloud-weighting term is needed.
* To try a different a priori (MUSICA / GCHP / GEOS-CF) just pass a different
  `apriori_profile` + `apriori_edges_hPa`.

This module is deliberately GENERAL: it is the recalculation core only. Reading
the L2 files, matching the two instruments' pixels, and any gridding are left to
the caller (see tropomi_regrid_l2_to_l3.py for an example L2 reader via OPeNDAP).

References
----------
  TROPOMI NO2 Product User Manual   S5P-KNMI-L2-0021-MA
  TROPOMI NO2 ATBD                  S5P-KNMI-L2-0005-RP
"""
import numpy as np


def mass_conservative_sigma_interp(src_edges, src_profile, tgt_edges):
    """Re-bin a partial-column profile from one layer grid onto another,
    conserving the column within the overlap, in sigma = P / Ps coordinates
    (surface-pressure invariant).

    Parameters
    ----------
    src_edges   : (n_src + 1,) sigma edges of the source layers (any orientation).
    src_profile : (n_src,) partial column in each source layer [molec cm-2].
    tgt_edges   : (n_tgt + 1,) sigma edges of the target layers.

    Returns
    -------
    (n_tgt,) partial column on the target layers.
    """
    src_edges = np.asarray(src_edges, float)
    tgt_edges = np.asarray(tgt_edges, float)
    src_lo = np.maximum(src_edges[:-1], src_edges[1:])   # larger sigma (nearer surface)
    src_hi = np.minimum(src_edges[:-1], src_edges[1:])   # smaller sigma (nearer TOA)
    src_thick = src_lo - src_hi
    out = np.zeros(len(tgt_edges) - 1)
    for j in range(len(out)):
        t_lo = max(tgt_edges[j], tgt_edges[j + 1])
        t_hi = min(tgt_edges[j], tgt_edges[j + 1])
        overlap = np.clip(np.minimum(src_lo, t_lo) - np.maximum(src_hi, t_hi), 0.0, None)
        frac = np.divide(overlap, src_thick, out=np.zeros_like(src_thick), where=src_thick > 0)
        out[j] = np.nansum(src_profile * frac)
    return out


def recalc_tropospheric_vcd(vcd_trop, ak_trop, layer_edges_hPa, tropopause_hPa,
                            apriori_profile, apriori_edges_hPa, surface_pressure_hPa):
    """Recompute a tropospheric VCD under a new a priori profile (ATBD Eq. 4).

    Parameters
    ----------
    vcd_trop            : retrieved tropospheric VCD [molec cm-2].
    ak_trop             : (n_layer,) tropospheric averaging kernel on the
                          retrieval grid (stratosphere already zeroed).
    layer_edges_hPa     : (n_layer + 1,) edge pressures of the retrieval layers.
    tropopause_hPa      : tropopause pressure — use the a priori model's own.
    apriori_profile     : (n_src,) NEW a priori partial columns [molec cm-2].
    apriori_edges_hPa   : (n_src + 1,) edge pressures of the a priori layers.
    surface_pressure_hPa: surface pressure.

    Returns
    -------
    vcd_trop_recalc [molec cm-2], or NaN if the tropospheric AK-weighted sum is
    non-positive.
    """
    layer_edges_hPa = np.asarray(layer_edges_hPa, float)
    ak_trop = np.asarray(ak_trop, float)

    # interpolate the a priori onto the retrieval grid in sigma space
    x_new = mass_conservative_sigma_interp(
        apriori_edges_hPa / surface_pressure_hPa,
        apriori_profile,
        layer_edges_hPa / surface_pressure_hPa,
    )

    # tropospheric layers = below the tropopause (larger pressure)
    layer_lo = np.maximum(layer_edges_hPa[:-1], layer_edges_hPa[1:])
    trop = layer_lo >= tropopause_hPa

    num = np.nansum(np.where(trop, x_new, 0.0))
    den = np.nansum(np.where(trop, ak_trop * x_new, 0.0))
    return vcd_trop * (num / den) if den > 0 else np.nan


if __name__ == "__main__":
    # Minimal illustrative example (synthetic numbers, not physical): swap an
    # a priori and see the recalculated column. Replace the arrays with a real
    # TM5 AK/grid and your model's (MUSICA / GCHP / GEOS-CF) partial columns.
    ps = 1013.0
    layer_edges = np.linspace(ps, 1.0, 35)                 # 34 retrieval layers
    ak = np.linspace(1.3, 0.2, 34)                          # toy tropospheric AK
    ap_edges = np.linspace(ps, 1.0, 73)                     # 72 a priori layers
    ap_prof = np.exp(-np.linspace(0, 4, 72)) * 1e15         # toy partial columns
    out = recalc_tropospheric_vcd(vcd_trop=5e15, ak_trop=ak,
                                  layer_edges_hPa=layer_edges, tropopause_hPa=150.0,
                                  apriori_profile=ap_prof, apriori_edges_hPa=ap_edges,
                                  surface_pressure_hPa=ps)
    print(f"recalculated VCD_trop = {out:.3e} molec cm-2")
