#!/usr/bin/env python
"""Build the TPXO9 reference, amplitude-bias and complex-RMSE maps for the
3D baroclinic runs shown on tpxo.html.

The 3D runs were produced on a machine without TPXO9, so the dashboard only
ever had bare model amplitude maps for them. This reads TPXO9-atlas v5
directly, samples it onto the MPAS mesh, and writes three figure families:

    img/tpxo/tpxo9/<C>_amp.png        TPXO9 amplitude          (run-independent)
    img/tpxo/<run>/<C>_bias.png       Am - At                  (per run)
    img/tpxo/<run>/<C>_rmse.png       complex RMSE Ec          (per run)

The amplitude figures deliberately reproduce the colour construction already
used by the model amplitude plots -- contourf over linspace(0, AMAX, 10) with
Spectral_r and extend='max' -- so the TPXO map can be read against them
directly. The error scales are tied to the same AMAX (+/-AMAX/10 for bias,
0..AMAX/5 for RMSE) and are shared across runs so the runs are comparable.

Complex RMSE follows the definition printed at the top of tpxo.html:

    Ec = sqrt[ 1/2 (Am^2 + At^2) - Am At cos(phim - phit) ]

Run it with the E3SM unified environment:

    source /global/common/software/e3sm/anaconda_envs/load_e3sm_unified_1.13.0_pm-cpu.sh
    python tools/make_tpxo_plots.py

Takes a few minutes; writes 75 PNGs and tools/tpxo_metrics.json.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# paths

SCRATCH = "/pscratch/sd/w/wpringle"
TPXO_DIR = f"{SCRATCH}/tpxo9-atlas"
E3SM = f"{SCRATCH}/e3sm_scratch/pm-cpu"
SAVED = f"{E3SM}/IcoswISC30E3r5/saved_runs"

# Any restart on this mesh carries the mesh fields; all runs share
# IcoswISC30E3r5, so one file serves every run.
MESH = f"{E3SM}/IcoswISC30E3r5_JSL/run/IcoswISC30E3r5_JSL.mpaso.rst.0001-03-08_00000.nc"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(REPO, "img", "tpxo")

# run key -> (harmonicAnalysis file, label used in figure titles)
# Keys match runDirs in tpxo.html. Verified against the published means in
# saved_runs/README_valid_45day_suite.txt and against the existing plot titles.
RUNS = {
    "betaSAL-45d": (
        f"{SAVED}/run_betaSAL_constCd_66day/hist/"
        "IcoswISC30E3r5_betaSAL.mpaso.hist.am.harmonicAnalysis.0001-03-01.nc",
        "β SAL, const Cᵈ (45d)",
    ),
    "SAL29-45d": (
        f"{SAVED}/run_SAL29_constCd_66day/hist/"
        "IcoswISC30E3r5.mpaso.hist.am.harmonicAnalysis.0001-03-01.nc",
        "full SAL n=29, const Cᵈ (45d)",
    ),
    "SAL29-loglaw-45d": (
        f"{SAVED}/run_SAL29_loglaw_66day/hist/"
        "IcoswISC30E3r5_loglaw.mpaso.hist.am.harmonicAnalysis.0001-03-01.nc",
        "full SAL n=29, log-law Cᵈ (45d)",
    ),
    "SAL29-loglaw-dt600-45d": (
        f"{SAVED}/run_SAL29_loglaw_dt600_66day/hist/"
        "IcoswISC30E3r5_loglaw.mpaso.hist.am.harmonicAnalysis.0001-03-01.nc",
        "full SAL n=29, log-law Cᵈ, Δt=600 s (45d)",
    ),
    "SAL29-loglaw-powerlaw-45d": (
        f"{SAVED}/run_SAL29_loglaw_powerlaw_66day/hist/"
        "IcoswISC30E3r5_loglaw.mpaso.hist.am.harmonicAnalysis.0001-03-01.nc",
        "log-law Cᵈ + S&McW velocity filter (45d)",
    ),
    "JSL-drag-66d": (
        f"{SAVED}/run_JSL_singleband_66day/hist/"
        "IcoswISC30E3r5_JSL.mpaso.hist.am.harmonicAnalysis.0001-03-01.nc",
        "S&McW filter + JSL internal tide drag (45d)",
    ),
    "JSL-bandpass-66d": (
        f"{E3SM}/IcoswISC30E3r5_JSL/run/"
        "IcoswISC30E3r5_JSL.mpaso.hist.am.harmonicAnalysis.0001-03-01.nc",
        "JSL drag split into semidiurnal/diurnal bands (45d)",
    ),
}

CONSTITUENTS = ["M2", "S2", "N2", "K1", "O1"]

# Upper bound of the amplitude colour scale, per constituent. Read off the
# existing model amplitude plots; the bias and RMSE scales derive from these.
AMAX = {"M2": 1.40, "S2": 0.60, "N2": 0.30, "K1": 0.60, "O1": 0.40}

# --------------------------------------------------------------------------
# figure geometry, measured off the existing model amplitude plots so the new
# figures stack against them without jumping

CANVAS = (9.30, 4.90)  # inches at dpi=100 -> 930 x 490 px
DPI = 100
MAP_AXES = [0.0140, 0.0571, 0.8742, 0.8306]
CBAR_AXES = [0.9108, 0.0571, 0.0204, 0.8306]
LAND_COLOR = "#E9E9DD"

GRID_RES = 0.25  # degrees; ~ the 30 km mesh, so TPXO is not drawn artificially sharp


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# mesh and regridding


def load_mesh():
    m = xr.open_dataset(MESH, decode_timedelta=False)
    lon = np.degrees(m.lonCell.values) % 360.0
    lat = np.degrees(m.latCell.values)
    return {
        "lon": lon,
        "lat": lat,
        "depth": m.bottomDepth.values,
        "area": m.areaCell.values,
    }


def build_regridder(mesh):
    """Nearest MPAS cell for each point of a regular lon/lat grid.

    Done in 3D Cartesian space so the dateline and the poles need no special
    casing. Built once and reused for every field.
    """
    glon = np.arange(-180.0, 180.0 + GRID_RES, GRID_RES)
    glat = np.arange(-90.0, 90.0 + GRID_RES, GRID_RES)
    LON, LAT = np.meshgrid(glon, glat)

    def xyz(lon_deg, lat_deg):
        la, lo = np.radians(lat_deg), np.radians(lon_deg)
        return np.column_stack(
            [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
        )

    tree = cKDTree(xyz(mesh["lon"], mesh["lat"]))
    _, idx = tree.query(xyz(LON.ravel(), LAT.ravel()), k=1)
    return glon, glat, idx.reshape(LON.shape)


def to_grid(cell_values, cell_mask, gidx):
    """Map per-cell values onto the regular grid, masking invalid cells."""
    out = np.ma.masked_array(cell_values[gidx], mask=~cell_mask[gidx])
    return out


# --------------------------------------------------------------------------
# TPXO


def tpxo_at_cells(constituent, mesh):
    """Nearest-neighbour sample of TPXO9-atlas onto MPAS cell centres.

    hRe/hIm are int32 millimetres; land is flagged by both being exactly zero.
    GMT phase convention, per the file's own `field` attribute:
    amp = |hRe + i hIm|, phase = atan2(-hIm, hRe).
    """
    path = f"{TPXO_DIR}/h_{constituent.lower()}_tpxo9_atlas_30_v5.nc"
    with xr.open_dataset(path) as t:
        lon_z = t.lon_z.values
        lat_z = t.lat_z.values
        hre = t.hRe.values
        him = t.hIm.values

    dlon = lon_z[1] - lon_z[0]
    dlat = lat_z[1] - lat_z[0]
    ix = np.round((mesh["lon"] - lon_z[0]) / dlon).astype(int) % lon_z.size
    iy = np.clip(np.round((mesh["lat"] - lat_z[0]) / dlat).astype(int), 0, lat_z.size - 1)

    re = hre[ix, iy]
    im = him[ix, iy]
    ocean = ~((re == 0) & (im == 0))
    re = re.astype(np.float64) / 1000.0
    im = im.astype(np.float64) / 1000.0
    amp = np.hypot(re, im)
    phase = np.degrees(np.arctan2(-im, re)) % 360.0
    return amp, phase, ocean


# --------------------------------------------------------------------------
# plotting


def base_figure(title):
    fig = plt.figure(figsize=CANVAS, dpi=DPI)
    ax = fig.add_axes(MAP_AXES, projection=ccrs.PlateCarree())
    ax.set_global()
    fig.text(0.45, 0.975, title, ha="center", va="top", fontsize=11, linespacing=1.35)
    return fig, ax


def finish(fig, ax, out):
    ax.add_feature(
        cfeature.LAND, facecolor=LAND_COLOR, edgecolor="black", linewidth=0.4, zorder=3
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_amplitude(glon, glat, field, phase, constituent, title, out):
    """Amplitude map in the exact style of the model amplitude plots."""
    levels = np.linspace(0.0, AMAX[constituent], 10)
    fig, ax = base_figure(title)
    cf = ax.contourf(
        glon,
        glat,
        field,
        levels=levels,
        cmap="Spectral_r",
        extend="max",
        transform=ccrs.PlateCarree(),
    )
    if phase is not None:
        # co-tidal lines every 30 deg, as on the model plots
        ax.contour(
            glon,
            glat,
            phase,
            levels=np.arange(0, 360, 30),
            colors="black",
            linewidths=0.25,
            transform=ccrs.PlateCarree(),
        )
    cax = fig.add_axes(CBAR_AXES)
    cb = fig.colorbar(cf, cax=cax, ticks=levels, format="%.2f")
    cb.ax.tick_params(labelsize=9)
    finish(fig, ax, out)


def plot_diverging(glon, glat, field, vmax, title, out):
    levels = np.linspace(-vmax, vmax, 17)
    fig, ax = base_figure(title)
    cf = ax.contourf(
        glon,
        glat,
        field,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    cax = fig.add_axes(CBAR_AXES)
    cb = fig.colorbar(cf, cax=cax, ticks=levels[::2], format="%.1f")
    cb.ax.tick_params(labelsize=9)
    finish(fig, ax, out)


def plot_sequential(glon, glat, field, vmax, title, out):
    levels = np.linspace(0.0, vmax, 11)
    fig, ax = base_figure(title)
    cf = ax.contourf(
        glon,
        glat,
        field,
        levels=levels,
        cmap="Reds",
        extend="max",
        transform=ccrs.PlateCarree(),
    )
    cax = fig.add_axes(CBAR_AXES)
    cb = fig.colorbar(cf, cax=cax, ticks=levels, format="%.1f")
    cb.ax.tick_params(labelsize=9)
    finish(fig, ax, out)


# --------------------------------------------------------------------------
# metrics


def region_masks(mesh, valid):
    """The three regions defined at the top of tpxo.html."""
    depth, lat = mesh["depth"], mesh["lat"]
    return {
        "global": valid & (depth > 20),
        "deep": valid & (depth >= 1000) & (np.abs(lat) < 66),
        "shallow": valid & (depth > 20) & (depth < 1000) & (np.abs(lat) < 66),
    }


def area_weighted_rms(values, weights):
    return float(np.sqrt(np.average(values**2, weights=weights)))


# --------------------------------------------------------------------------


def main():
    log("loading mesh")
    mesh = load_mesh()
    log(f"  {mesh['lon'].size} cells")

    log("building regridder")
    glon, glat, gidx = build_regridder(mesh)
    log(f"  grid {glon.size} x {glat.size}")

    metrics = {}

    for c in CONSTITUENTS:
        log(f"\n=== {c}")
        at, pt, tpxo_ocean = tpxo_at_cells(c, mesh)

        # Amplitude maps show the ocean as the model plots do; the error maps
        # additionally drop water shallower than 20 m, matching the metric
        # definition on the page.
        amp_valid = tpxo_ocean & np.isfinite(at)
        err_valid = amp_valid & (mesh["depth"] > 20)

        # --- TPXO reference, run-independent
        out = os.path.join(IMG, "tpxo9", f"{c}_amp.png")
        w = mesh["area"][err_valid]
        title = (
            f"{c} Amplitude (TPXO9-atlas v5) [m]\n"
            f"sampled to IcoswISC30E3r5 cells   "
            f"mean {at[err_valid].mean():.3f} m, max {at[amp_valid].max():.2f} m"
        )
        plot_amplitude(
            glon,
            glat,
            to_grid(at, amp_valid, gidx),
            to_grid(pt, amp_valid, gidx),
            c,
            title,
            out,
        )
        log(f"  wrote {os.path.relpath(out, REPO)}")

        for run, (path, label) in RUNS.items():
            with xr.open_dataset(path, decode_timedelta=False) as h:
                am = h[f"{c}Amplitude"].values[0].astype(np.float64)
                pm = h[f"{c}Phase"].values[0].astype(np.float64)

            valid = err_valid & np.isfinite(am) & np.isfinite(pm)
            bias = am - at
            dphi = np.radians(pm - pt)
            ec = np.sqrt(
                np.maximum(0.5 * (am**2 + at**2) - am * at * np.cos(dphi), 0.0)
            )

            regions = region_masks(mesh, valid)
            mrec = {}
            for rname, rmask in regions.items():
                w = mesh["area"][rmask]
                mrec[rname] = {
                    "rmse_cm": 100.0 * area_weighted_rms(ec[rmask], w),
                    "bias_cm": 100.0 * float(np.average(bias[rmask], weights=w)),
                }
            metrics.setdefault(run, {})[c] = mrec

            g = mrec["global"]
            bias_out = os.path.join(IMG, run, f"{c}_bias.png")
            plot_diverging(
                glon,
                glat,
                to_grid(100.0 * bias, valid, gidx),
                100.0 * AMAX[c] / 10.0,
                f"{c} Amplitude bias — {label} [cm]\n"
                f"model − TPXO9   area-weighted mean {g['bias_cm']:+.2f} cm",
                bias_out,
            )

            rmse_out = os.path.join(IMG, run, f"{c}_rmse.png")
            plot_sequential(
                glon,
                glat,
                to_grid(100.0 * ec, valid, gidx),
                100.0 * AMAX[c] / 5.0,
                f"{c} Complex RMSE — {label} [cm]\n"
                f"area-weighted: global {g['rmse_cm']:.2f}, "
                f"deep {mrec['deep']['rmse_cm']:.2f}, "
                f"shallow {mrec['shallow']['rmse_cm']:.2f} cm",
                rmse_out,
            )
            log(
                f"  {run:26s} Ec {g['rmse_cm']:6.2f} cm   bias {g['bias_cm']:+6.2f} cm"
            )

    mpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpxo_metrics.json")
    with open(mpath, "w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    log(f"\nwrote {os.path.relpath(mpath, REPO)}")


if __name__ == "__main__":
    main()
