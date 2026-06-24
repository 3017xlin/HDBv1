"""MAD-based anomaly detection for HDB CFD cases.

19 detectors (9 mean + 10 std):
  Mean detectors: Ux, Uy, Uz, p_vol, nut_nondim, p_surf, wss_x, wss_y, wss_z
  Std detectors:  Ux, Uy, Uz, p_vol, log_nut, p_surf, wss_x, wss_y, wss_z, nut_nondim

Rules:
  Rule 1 (concordance): >= 2 different fields each have z_MAD > 5.0
  Rule 2 (extreme):     any single z_MAD > 8.0

Operates on PHYSICAL PT values BEFORE any transforms.  The caller computes
per-case field statistics (mean / std of each field) and passes them in.
"""
from __future__ import annotations

import numpy as np

# ── 19 indicator names ─────────────────────────────────────────────
# Convention: "<field>_mean" or "<field>_std"
INDICATOR_NAMES: list[str] = [
    # 9 mean detectors
    "Ux_mean",
    "Uy_mean",
    "Uz_mean",
    "p_vol_mean",
    "nut_nondim_mean",
    "p_surf_mean",
    "wss_x_mean",
    "wss_y_mean",
    "wss_z_mean",
    # 10 std detectors
    "Ux_std",
    "Uy_std",
    "Uz_std",
    "p_vol_std",
    "log_nut_std",
    "p_surf_std",
    "wss_x_std",
    "wss_y_std",
    "wss_z_std",
    "nut_nondim_std",
]

# Map each indicator to its parent "field" for concordance counting.
# Two indicators that share a field count as one field hit.
_FIELD_MAP: dict[str, str] = {}
for _name in INDICATOR_NAMES:
    # Strip trailing _mean / _std to get field name
    if _name.endswith("_mean"):
        _FIELD_MAP[_name] = _name[: -len("_mean")]
    elif _name.endswith("_std"):
        _FIELD_MAP[_name] = _name[: -len("_std")]

CONCORDANCE_THRESHOLD: float = 5.0
EXTREME_THRESHOLD: float = 8.0
MAD_SCALE: float = 1.4826  # Gaussian consistency factor


def detect_anomalies_mad(
    cases_stats: list[dict],
) -> dict[str, dict]:
    """Run MAD-based anomaly detection across all cases.

    Parameters
    ----------
    cases_stats : list[dict]
        Each dict must contain:
          - ``"case_name"`` : str
          - One key per entry in :pydata:`INDICATOR_NAMES` holding a float
            value (the per-case statistic computed from the physical PT).

    Returns
    -------
    dict[str, dict]
        Keyed by case_name.  Each value is::

            {
                "is_anomaly": bool,
                "rule": "concordance_2" | "extreme_single" | "normal",
                "hits": [{"indicator": ..., "z_mad": ...}, ...],
            }
    """
    if len(cases_stats) == 0:
        return {}

    n_cases = len(cases_stats)
    n_ind = len(INDICATOR_NAMES)

    # Build (N_cases, 19) matrix
    values = np.empty((n_cases, n_ind), dtype=np.float64)
    for i, cs in enumerate(cases_stats):
        for j, name in enumerate(INDICATOR_NAMES):
            values[i, j] = cs[name]

    # MAD z-scores
    medians = np.median(values, axis=0)  # (19,)
    abs_dev = np.abs(values - medians)  # (N_cases, 19)
    mad_scaled = MAD_SCALE * np.median(abs_dev, axis=0)  # (19,)
    mad_scaled = np.maximum(mad_scaled, 1e-10)
    z_mad = abs_dev / mad_scaled  # (N_cases, 19)

    results: dict[str, dict] = {}
    for i, cs in enumerate(cases_stats):
        z = z_mad[i]

        # Per-field maximum z_MAD (merge mean and std of same field)
        field_max_z: dict[str, float] = {}
        for j, name in enumerate(INDICATOR_NAMES):
            field = _FIELD_MAP[name]
            field_max_z[field] = max(field_max_z.get(field, 0.0), z[j])

        flagged_fields = [f for f, fz in field_max_z.items() if fz > CONCORDANCE_THRESHOLD]
        extreme_any = bool(np.any(z > EXTREME_THRESHOLD))

        is_anomaly = len(flagged_fields) >= 2 or extreme_any
        if len(flagged_fields) >= 2:
            rule = "concordance_2"
        elif extreme_any:
            rule = "extreme_single"
        else:
            rule = "normal"

        # Collect detailed hits for diagnostics
        hits = []
        for j, name in enumerate(INDICATOR_NAMES):
            if z[j] > CONCORDANCE_THRESHOLD:
                hits.append({"indicator": name, "z_mad": float(z[j])})

        results[cs["case_name"]] = {
            "is_anomaly": is_anomaly,
            "rule": rule,
            "hits": hits,
        }

    return results


def compute_case_stats(pt: dict) -> dict:
    """Compute the 19 indicator values from a single physical PT dict.

    This is a convenience helper so that callers don't have to hand-roll the
    indicator extraction.  ``pt`` should contain the raw (pre-transform)
    tensors as loaded from the physical PT file.

    Returns a dict suitable for inclusion in the ``cases_stats`` list expected
    by :func:`detect_anomalies_mad`.
    """
    import torch

    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.float().numpy()
        return np.asarray(x, dtype=np.float32)

    # Volume fields: [Ux, Uy, Uz, p, nut]  shape (N_vol, 5)
    vf = _t(pt["volume_fields"])
    Ux, Uy, Uz = vf[:, 0], vf[:, 1], vf[:, 2]
    p_vol = vf[:, 3]
    nut = vf[:, 4]

    # Surface fields: [p, wss_x, wss_y, wss_z]  shape (N_surf, 4)
    sf = _t(pt["surface_fields"])
    p_surf = sf[:, 0]
    wss_x, wss_y, wss_z = sf[:, 1], sf[:, 2], sf[:, 3]

    # Derived
    nut_nondim = nut  # raw physical nut values
    log_nut = np.log(np.maximum(np.abs(nut), 1e-30))

    stats: dict = {
        "case_name": pt.get("case_name", "unknown"),
        # 9 mean detectors
        "Ux_mean": float(np.mean(Ux)),
        "Uy_mean": float(np.mean(Uy)),
        "Uz_mean": float(np.mean(Uz)),
        "p_vol_mean": float(np.mean(p_vol)),
        "nut_nondim_mean": float(np.mean(nut_nondim)),
        "p_surf_mean": float(np.mean(p_surf)),
        "wss_x_mean": float(np.mean(wss_x)),
        "wss_y_mean": float(np.mean(wss_y)),
        "wss_z_mean": float(np.mean(wss_z)),
        # 10 std detectors
        "Ux_std": float(np.std(Ux)),
        "Uy_std": float(np.std(Uy)),
        "Uz_std": float(np.std(Uz)),
        "p_vol_std": float(np.std(p_vol)),
        "log_nut_std": float(np.std(log_nut)),
        "p_surf_std": float(np.std(p_surf)),
        "wss_x_std": float(np.std(wss_x)),
        "wss_y_std": float(np.std(wss_y)),
        "wss_z_std": float(np.std(wss_z)),
        "nut_nondim_std": float(np.std(nut_nondim)),
    }
    return stats
