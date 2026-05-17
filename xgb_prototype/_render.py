"""Raster rendering controller for large diagnostics.

The plotting module sends large scatter/heatmap payloads here and does not need
to know whether RAPIDS/cuXfilter, Datashader, or a local fallback did the work.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .settings import PLOT_OUTPUT_DIR, RASTER_FORMAT, USE_GPU

log = logging.getLogger(__name__)

_CUXFILTER_CHECKED = False
_CUXFILTER_AVAILABLE = False
_DATASHADER_CHECKED = False
_DATASHADER_AVAILABLE = False


def _output_path(stem: str) -> Path:
    suffix = ".jpg" if RASTER_FORMAT == "jpeg" else f".{RASTER_FORMAT}"
    return PLOT_OUTPUT_DIR / f"{stem}{suffix}"


def _html_path(stem: str) -> Path:
    return PLOT_OUTPUT_DIR / f"{stem}.html"


def _check_cuxfilter() -> bool:
    global _CUXFILTER_CHECKED, _CUXFILTER_AVAILABLE
    if _CUXFILTER_CHECKED:
        return _CUXFILTER_AVAILABLE
    _CUXFILTER_CHECKED = True
    if not USE_GPU:
        return False
    try:
        __import__("cuxfilter")
        _CUXFILTER_AVAILABLE = True
    except Exception:
        log.info("  [render] USE_GPU=True but RAPIDS/cuXfilter is unavailable; using Datashader.")
        _CUXFILTER_AVAILABLE = False
    return _CUXFILTER_AVAILABLE


def _check_datashader() -> bool:
    global _DATASHADER_CHECKED, _DATASHADER_AVAILABLE
    if _DATASHADER_CHECKED:
        return _DATASHADER_AVAILABLE
    _DATASHADER_CHECKED = True
    try:
        __import__("datashader")
        __import__("holoviews")
        _DATASHADER_AVAILABLE = True
    except Exception as exc:
        log.warning("  [render] Datashader/HoloViews unavailable; using Matplotlib fallback: %s", exc)
        _DATASHADER_AVAILABLE = False
    return _DATASHADER_AVAILABLE


def _save_matplotlib(fig, path: Path) -> None:
    save_kwargs = {"bbox_inches": "tight"}
    if RASTER_FORMAT == "jpeg":
        save_kwargs["format"] = "jpeg"
        save_kwargs["facecolor"] = "white"
    else:
        save_kwargs["format"] = RASTER_FORMAT
    fig.savefig(str(path), dpi=130, **save_kwargs)


def _limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        pad = max(abs(lo) * 0.01, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.03
    return lo - pad, hi + pad


def _numeric_color(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return numeric if np.isfinite(numeric).any() else None


def scatter_large(
    x: Iterable,
    y: Iterable,
    *,
    color: Iterable | None = None,
    title: str,
    x_label: str,
    y_label: str,
    output_name: str,
    width: int = 900,
    height: int = 620,
    color_label: str | None = None,
    y_tickvals: list[float] | None = None,
    y_ticktext: list[str] | None = None,
    cmap: str = "viridis",
) -> Path:
    """Render a large scatter to the configured raster format."""
    _check_cuxfilter()
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    color_arr = None if color is None else np.asarray(list(color), dtype=object)[mask]
    color_num = _numeric_color(color_arr)
    path = _output_path(output_name)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w, fig_h = max(5.0, width / 140), max(3.5, height / 140)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if _check_datashader() and len(x_arr) > 5_000:
        import datashader as ds
        import datashader.transfer_functions as tf

        df = pd.DataFrame({"x": x_arr, "y": y_arr})
        x_range, y_range = _limits(x_arr), _limits(y_arr)
        canvas = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
        if color_num is not None:
            df["color"] = color_num
            agg = canvas.points(df, "x", "y", ds.mean("color"))
            img = tf.shade(agg, cmap=plt.get_cmap(cmap))
        elif color_arr is not None:
            df["color"] = pd.Categorical(color_arr.astype(str))
            cats = list(df["color"].cat.categories)
            palette = plt.get_cmap("tab10")
            color_key = {cat: matplotlib.colors.to_hex(palette(i % 10)) for i, cat in enumerate(cats)}
            agg = canvas.points(df, "x", "y", ds.count_cat("color"))
            img = tf.shade(agg, color_key=color_key)
        else:
            agg = canvas.points(df, "x", "y", ds.count())
            img = tf.shade(agg, cmap=plt.get_cmap(cmap))
        img = tf.dynspread(img, max_px=2)
        ax.imshow(np.asarray(img.to_pil()), extent=(*x_range, *y_range), origin="upper", aspect="auto")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
    else:
        if color_num is not None:
            sc = ax.scatter(x_arr, y_arr, c=color_num, cmap=cmap, s=8, alpha=0.65,
                            linewidths=0, rasterized=True)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
            if color_label:
                cbar.set_label(color_label)
        elif color_arr is not None:
            vals = pd.Categorical(color_arr.astype(str))
            palette = plt.get_cmap("tab10")
            for idx, cat in enumerate(vals.categories):
                cat_mask = vals == cat
                ax.scatter(x_arr[cat_mask], y_arr[cat_mask], s=8, alpha=0.62,
                           linewidths=0, color=palette(idx % 10), label=cat, rasterized=True)
            ax.legend(title=color_label or "", fontsize=8, markerscale=2)
        else:
            ax.scatter(x_arr, y_arr, s=8, alpha=0.65, linewidths=0, rasterized=True)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_tickvals is not None and y_ticktext is not None:
        ax.set_yticks(y_tickvals)
        ax.set_yticklabels(y_ticktext)
    ax.spines[["top", "right"]].set_visible(False)
    _save_matplotlib(fig, path)
    plt.close(fig)
    return path


def scatter_large_html(
    x: Iterable,
    y: Iterable,
    *,
    color: Iterable | None = None,
    title: str,
    x_label: str,
    y_label: str,
    output_name: str,
    width: int = 900,
    height: int = 620,
    color_label: str | None = None,
    cmap: str = "viridis",
) -> Path:
    """Render a large scatter as Datashader/HoloViews HTML."""
    _check_cuxfilter()
    if not _check_datashader():
        return scatter_large(
            x, y, color=color, title=title, x_label=x_label, y_label=y_label,
            output_name=output_name, width=width, height=height,
            color_label=color_label, cmap=cmap,
        )

    import datashader as ds
    import holoviews as hv
    from holoviews.operation.datashader import datashade, dynspread, rasterize

    hv.extension("bokeh", logo=False)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    df = pd.DataFrame({x_label: x_arr, y_label: y_arr})
    color_arr = None if color is None else np.asarray(list(color), dtype=object)[mask]
    color_name = color_label or "value"
    color_num = _numeric_color(color_arr)

    if color_num is not None:
        df[color_name] = color_num
        points = hv.Points(df, kdims=[x_label, y_label], vdims=[color_name])
        plot = rasterize(points, aggregator=ds.mean(color_name), width=width, height=height).opts(
            cmap=cmap, colorbar=True, tools=["hover"], title=title,
            width=width, height=height, xlabel=x_label, ylabel=y_label,
        )
    elif color_arr is not None:
        df[color_name] = pd.Categorical(color_arr.astype(str))
        cats = list(df[color_name].cat.categories)
        palette = ["#4477AA", "#CC3333", "#44AA77", "#AA3377", "#BBBB44", "#66CCEE"]
        color_key = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}
        points = hv.Points(df, kdims=[x_label, y_label], vdims=[color_name])
        plot = dynspread(datashade(
            points, aggregator=ds.count_cat(color_name), color_key=color_key,
            width=width, height=height,
        )).opts(title=title, width=width, height=height, xlabel=x_label, ylabel=y_label)
    else:
        points = hv.Points(df, kdims=[x_label, y_label])
        plot = dynspread(datashade(points, width=width, height=height, cmap=cmap)).opts(
            title=title, width=width, height=height, xlabel=x_label, ylabel=y_label,
        )

    path = _html_path(output_name)
    hv.save(plot, str(path), backend="bokeh")
    return path


def heatmap_large(
    z: Iterable,
    *,
    title: str,
    x_label: str = "",
    y_label: str = "",
    output_name: str,
    x_ticktext: list[str] | None = None,
    y_ticktext: list[str] | None = None,
    cmap: str = "viridis",
    center: float | None = None,
    value_label: str | None = None,
    annotate: bool = False,
    show_grid: bool = False,
    width: int = 900,
    height: int = 700,
) -> Path:
    """Render a matrix-like diagnostic to the configured raster format."""
    _check_cuxfilter()
    arr = np.asarray(z, dtype=float)
    path = _output_path(output_name)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig_w, fig_h = max(5.0, width / 140), max(3.5, height / 140)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    norm = None
    finite = arr[np.isfinite(arr)]
    if center is not None and finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin < center < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    # Datashader is used as the default raster path for larger matrices. The
    # Matplotlib imshow call below only places the already-raster image on axes.
    if _check_datashader() and arr.size > 10_000:
        import xarray as xr
        import datashader.transfer_functions as tf

        da = xr.DataArray(np.nan_to_num(arr, nan=0.0), dims=["y", "x"])
        img = tf.shade(da, cmap=plt.get_cmap(cmap))
        ax.imshow(np.asarray(img.to_pil()), aspect="auto")
    else:
        im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        if value_label:
            cbar.set_label(value_label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_ticktext is not None:
        ax.set_xticks(np.arange(len(x_ticktext)))
        ax.set_xticklabels(x_ticktext, rotation=35, ha="right")
    else:
        ax.set_xticks([])
    if y_ticktext is not None:
        ax.set_yticks(np.arange(len(y_ticktext)))
        ax.set_yticklabels(y_ticktext)
    else:
        ax.set_yticks([])
    if show_grid:
        ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)
    if annotate and arr.size <= 2_500:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    _save_matplotlib(fig, path)
    plt.close(fig)
    return path


def heatmap_large_html(
    z: Iterable,
    *,
    title: str,
    output_name: str,
    x_ticktext: list[str],
    y_ticktext: list[str],
    cmap: str = "RdBu_r",
    center: float | None = None,
    value_label: str = "value",
    width: int = 900,
    height: int = 700,
) -> Path:
    """Render a matrix as HoloViews HTML with hoverable cells."""
    _check_cuxfilter()
    if not _check_datashader():
        return heatmap_large(
            z, title=title, output_name=output_name, x_ticktext=x_ticktext,
            y_ticktext=y_ticktext, cmap=cmap, center=center,
            value_label=value_label, width=width, height=height,
        )

    import holoviews as hv

    hv.extension("bokeh", logo=False)
    arr = np.asarray(z, dtype=float)
    rows = []
    for yi, y_name in enumerate(y_ticktext):
        for xi, x_name in enumerate(x_ticktext):
            val = arr[yi, xi]
            if np.isfinite(val):
                rows.append({"x": x_name, "y": y_name, value_label: float(val)})
    df = pd.DataFrame(rows)
    clim = None
    finite = arr[np.isfinite(arr)]
    if center is not None and finite.size:
        bound = float(np.max(np.abs(finite - center)))
        clim = (center - bound, center + bound)
    plot = hv.HeatMap(df, kdims=["x", "y"], vdims=[value_label]).opts(
        title=title,
        cmap=cmap,
        clim=clim,
        colorbar=True,
        tools=["hover"],
        width=width,
        height=height,
        xrotation=45,
        invert_yaxis=True,
        labelled=[],
        line_color="white",
        line_width=0.5,
    )
    path = _html_path(output_name)
    hv.save(plot, str(path), backend="bokeh")
    return path
