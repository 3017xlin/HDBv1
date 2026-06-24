"""Visualization: horizontal slices and error distribution bar charts.

Generates per-height-slice plots for the median-error test case and an
aggregate bar chart of per-field relative L2 across all test cases.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array to float32 numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Horizontal slice plots
# ---------------------------------------------------------------------------

def plot_horizontal_slices(
    vol_pos: np.ndarray | torch.Tensor,
    gt_fields: dict[str, np.ndarray | torch.Tensor],
    pred_fields: dict[str, np.ndarray | torch.Tensor],
    case_name: str,
    out_dir: str | Path,
    slice_heights: Sequence[float] = (1.5, 5, 10, 20, 40, 80, 120),
    slice_tolerances: Sequence[float] = (0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 3.0),
) -> list[str]:
    """Plot GT / Pred / |Error| for velocity magnitude and pressure at each z slice.

    Parameters
    ----------
    vol_pos : (N_vol, 3) physical coordinates in metres.
    gt_fields : dict with 'Ux', 'Uy', 'Uz', 'p' arrays each (N_vol,).
    pred_fields : same keys.
    case_name : identifier for the case (used in titles).
    out_dir : directory to save PNG files.
    slice_heights : z values for horizontal slices (metres).
    slice_tolerances : per-slice thickness tolerance (metres).

    Returns
    -------
    List of saved PNG file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = _to_numpy(vol_pos)
    gt = {k: _to_numpy(v) for k, v in gt_fields.items()}
    pred = {k: _to_numpy(v) for k, v in pred_fields.items()}

    # Velocity magnitude
    gt_vmag = np.sqrt(gt["Ux"]**2 + gt["Uy"]**2 + gt["Uz"]**2)
    pred_vmag = np.sqrt(pred["Ux"]**2 + pred["Uy"]**2 + pred["Uz"]**2)

    saved_paths: list[str] = []

    for z_target, tol in zip(slice_heights, slice_tolerances):
        mask = np.abs(pos[:, 2] - z_target) <= tol
        n_pts = int(mask.sum())
        if n_pts < 10:
            continue

        x = pos[mask, 0]
        y = pos[mask, 1]

        # Triangulation for irregular scatter
        try:
            tri = mtri.Triangulation(x, y)
        except (RuntimeError, ValueError):
            continue

        # Fields at this slice
        fields_to_plot = [
            ("Velocity Magnitude", gt_vmag[mask], pred_vmag[mask]),
            ("Pressure", gt["p"][mask], pred["p"][mask]),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Case: {case_name} | z = {z_target}m "
                     f"({n_pts} points, tol={tol}m)", fontsize=14)

        for row_idx, (field_name, gt_slice, pred_slice) in enumerate(fields_to_plot):
            error_slice = np.abs(pred_slice - gt_slice)

            # Color limits from GT percentiles
            vmin = float(np.percentile(gt_slice, 0.5))
            vmax = float(np.percentile(gt_slice, 99.5))
            if abs(vmax - vmin) < 1e-10:
                vmin -= 0.5
                vmax += 0.5

            # Error color limits
            emax = float(np.percentile(error_slice, 99.5))
            if emax < 1e-10:
                emax = 1.0

            panels = [
                ("GT", gt_slice, vmin, vmax, "viridis"),
                ("Pred", pred_slice, vmin, vmax, "viridis"),
                ("|Error|", error_slice, 0.0, emax, "hot"),
            ]

            for col_idx, (title, data, lo, hi, cmap) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                tc = ax.tripcolor(tri, data, shading="flat",
                                  vmin=lo, vmax=hi, cmap=cmap)
                fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{field_name} - {title}", fontsize=10)
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_aspect("equal")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        png_name = f"slice_z{z_target:.1f}m_{case_name}.png"
        png_path = out_dir / png_name
        fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(png_path))

    return saved_paths


# ---------------------------------------------------------------------------
# Error distribution bar chart
# ---------------------------------------------------------------------------

def plot_error_distribution(
    per_case_metrics: dict[str, dict[str, float]],
    out_dir: str | Path,
    filename: str = "test_error_distribution.png",
) -> str:
    """Bar chart of per-field relative L2 across all test cases.

    Shows mean with std error bars for each field.

    Returns path to saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_keys = ["Ux_rel_L2", "Uy_rel_L2", "Uz_rel_L2",
                  "p_vol_rel_L2", "nut_rel_L2", "p_surf_rel_L2"]
    field_labels = ["Ux", "Uy", "Uz", "p (vol)", "nut", "p (surf)"]

    means = []
    stds = []
    for fk in field_keys:
        vals = [m[fk] for m in per_case_metrics.values()
                if fk in m and not isinstance(m[fk], (dict, list))]
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        else:
            means.append(0.0)
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(field_keys))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue",
                  edgecolor="black", alpha=0.85)

    # Annotate bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.002, f"{m:.4f}", ha="center", va="bottom",
                fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(field_labels, fontsize=11)
    ax.set_ylabel("Relative L2 Error", fontsize=12)
    ax.set_title("Per-Field Test Set Relative L2 Error (mean +/- std)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / filename
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


# ---------------------------------------------------------------------------
# Training / validation curve plot
# ---------------------------------------------------------------------------

def plot_train_val_curve(
    curve_data: dict[int, dict[str, float]],
    out_dir: str | Path,
    swa_start_epoch: int = 300,
    filename: str = "train_val_curve.png",
) -> str:
    """Plot train_eval and val z-score MSE curves over epochs.

    curve_data: {epoch: {'train_eval_vol_mse': ..., 'train_eval_surf_mse': ...,
                         'val_vol_mse': ..., 'val_surf_mse': ...}}

    Returns path to saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = sorted(curve_data.keys())
    te_vol = [curve_data[e].get("train_eval_vol_mse", 0) for e in epochs]
    te_surf = [curve_data[e].get("train_eval_surf_mse", 0) for e in epochs]
    v_vol = [curve_data[e].get("val_vol_mse", 0) for e in epochs]
    v_surf = [curve_data[e].get("val_surf_mse", 0) for e in epochs]

    fig, (ax_v, ax_s) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    ax_v.plot(epochs, te_vol, label="train_eval", color="C0", lw=1.5)
    ax_v.plot(epochs, v_vol, label="val", color="C3", lw=1.5)
    ax_v.set_title("Volume 5d z-score MSE")
    ax_v.legend()
    ax_v.axvline(swa_start_epoch, color="gray", ls="--", alpha=0.6,
                 label="SWA start")
    ax_v.set_xlabel("Epoch")
    ax_v.grid(alpha=0.3)

    ax_s.plot(epochs, te_surf, label="train_eval", color="C0", lw=1.5)
    ax_s.plot(epochs, v_surf, label="val", color="C3", lw=1.5)
    ax_s.set_title("Surface 1d z-score MSE")
    ax_s.legend()
    ax_s.axvline(swa_start_epoch, color="gray", ls="--", alpha=0.6,
                 label="SWA start")
    ax_s.set_xlabel("Epoch")
    ax_s.grid(alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / filename
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


# ---------------------------------------------------------------------------
# Helper: select median-error case
# ---------------------------------------------------------------------------

def select_median_case(per_case_metrics: dict[str, dict[str, float]],
                       key: str = "mean_rel_L2") -> str:
    """Return the case_id with median error by the given metric key."""
    items = sorted(per_case_metrics.items(), key=lambda x: x[1].get(key, 0))
    return items[len(items) // 2][0]


# ---------------------------------------------------------------------------
# Convenience: generate all viz for a test result
# ---------------------------------------------------------------------------

def generate_test_visualizations(
    test_result: dict[str, Any],
    out_dir: str | Path,
    cfg: dict,
) -> list[str]:
    """Generate all visualization artifacts for the test evaluation result.

    Expects test_result from test_full_inference with median case arrays.

    Returns list of all saved file paths.
    """
    out_dir = Path(out_dir)
    all_paths: list[str] = []

    per_case = test_result.get("per_case", {})
    median_id = test_result.get("median_case_id")

    # Error distribution
    scalar_metrics = {c: {k: v for k, v in m.items() if not k.startswith("_")}
                      for c, m in per_case.items()}
    all_paths.append(plot_error_distribution(scalar_metrics, out_dir))

    # Horizontal slices for median case
    if median_id and median_id in per_case:
        mc = per_case[median_id]
        if "_pred_vol_phys" in mc and "_gt_vol_phys" in mc and "_vol_pos" in mc:
            pred_vol = _to_numpy(mc["_pred_vol_phys"])
            gt_vol = _to_numpy(mc["_gt_vol_phys"])
            vol_pos = _to_numpy(mc["_vol_pos"])

            gt_fields = {
                "Ux": gt_vol[:, 0], "Uy": gt_vol[:, 1],
                "Uz": gt_vol[:, 2], "p": gt_vol[:, 3],
            }
            pred_fields = {
                "Ux": pred_vol[:, 0], "Uy": pred_vol[:, 1],
                "Uz": pred_vol[:, 2], "p": pred_vol[:, 3],
            }

            slice_heights = cfg.get("evaluation", {}).get(
                "slice_heights", [1.5, 5, 10, 20, 40, 80, 120])
            slice_tolerances = cfg.get("evaluation", {}).get(
                "slice_tolerances", [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 3.0])

            paths = plot_horizontal_slices(
                vol_pos, gt_fields, pred_fields,
                case_name=str(median_id),
                out_dir=out_dir / "slices",
                slice_heights=slice_heights,
                slice_tolerances=slice_tolerances,
            )
            all_paths.extend(paths)

    return all_paths
