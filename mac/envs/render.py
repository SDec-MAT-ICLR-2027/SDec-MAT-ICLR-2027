"""
Shared drawing toolkit for the MAC benchmark renderers (figure-quality).

Why this file exists
--------------------
The drone benchmarks (Pickup Consensus, Aerial Ordering, Charging Rendezvous)
tell one story: interchangeable drones over a field that must agree on WHICH of
them commits. Their figures belong in the same paper, so they share one visual
language — the same drone glyph, the same battery gauge, the same comm-link
styling, the same HUD. Everything reusable lives here; each env only composes
the scene.

Usage from an env
-----------------
    from . import render as dr       # lazy, inside the env's _render_kit()
    fig, ax = dr.stage(...)
    dr.drone(ax, x, y, ...)

Themes
------
    "paper" (default) : warm off-white, print-safe, what the paper figures use.
    "night"           : dark ops-screen look, better for slides/GIFs.
    "blueprint"       : cyan-on-navy schematic, for architecture-style figures.

Frame capture
-------------
`finish()` either shows the figure interactively or writes a PNG, depending on
whether a save directory was configured. `write_gif()` stitches the PNGs.
Nothing here imports matplotlib at module import time except the pyplot handle
inside functions, so a headless run that never renders pays nothing.
"""

from __future__ import annotations

import math
import os

import numpy as np


# ======================================================================
# Themes
# ======================================================================

_THEMES = {
    # Print-first: warm paper ground, ink-black text, muted saturated accents.
    "paper": dict(
        bg="#F5F2EC",
        field_top="#EFEDE7",
        field_bottom="#E4E7DC",
        panel="#FFFFFF",
        panel_edge="#D9D3C7",
        ink="#1B1A17",
        muted="#8B857A",
        grid="#DFD9CD",
        grid_major="#CDC5B5",
        comm="#2E86AB",
        comm_fill="#2E86AB",
        ok="#2E8B57",
        warn="#DE9B0B",
        bad="#C1443C",
        sensor="#E2703A",
        crate="#C89B5A",
        crate_edge="#8A6534",
        pad="#5B6472",
        pad_edge="#39404B",
        shadow="#000000",
        halo="#FFFFFF",
        agents=[
            "#2F6F9F", "#3E9E8F", "#7C5BA6", "#C7544F",
            "#4C7A34", "#B4762A", "#4F5D75", "#A8467F",
        ],
    ),
    # Dark ops screen: good for slides and animated GIFs.
    "night": dict(
        bg="#10141A",
        field_top="#141A22",
        field_bottom="#0C1016",
        panel="#18202B",
        panel_edge="#2A3644",
        ink="#E9ECF1",
        muted="#7C8899",
        grid="#1D2733",
        grid_major="#28374A",
        comm="#4FC3F7",
        comm_fill="#4FC3F7",
        ok="#4CD08A",
        warn="#F2C14E",
        bad="#F2635F",
        sensor="#FF9152",
        crate="#C79A5B",
        crate_edge="#7A5B2E",
        pad="#3A4655",
        pad_edge="#5A6B80",
        shadow="#000000",
        halo="#0B0E13",
        agents=[
            "#59A5FF", "#4ECDC4", "#B18CFF", "#FF7B72",
            "#9CCC65", "#FFB74D", "#90A4AE", "#F06292",
        ],
    ),
    # Schematic look: everything reads as a technical drawing.
    "blueprint": dict(
        bg="#0E2A47",
        field_top="#123457",
        field_bottom="#0B2340",
        panel="#123457",
        panel_edge="#3E6E9E",
        ink="#E8F4FF",
        muted="#8FB4D6",
        grid="#1B4470",
        grid_major="#2A5C90",
        comm="#7FE7FF",
        comm_fill="#7FE7FF",
        ok="#6FE39B",
        warn="#FFD166",
        bad="#FF8A80",
        sensor="#FFB870",
        crate="#9FC7E8",
        crate_edge="#D6ECFF",
        pad="#1B4470",
        pad_edge="#7FE7FF",
        shadow="#000000",
        halo="#0E2A47",
        agents=[
            "#7FE7FF", "#9FE8C8", "#C7B6FF", "#FFAFA8",
            "#C6E38A", "#FFD08A", "#AFC7DD", "#FFA8D4",
        ],
    ),
}


def theme(name="paper"):
    """Colour dict for `name`, falling back to the paper theme.

    The dict carries its own name under `_name` so drawing helpers can pick a
    matching terrain palette without every caller having to pass it along.
    """
    th = dict(_THEMES.get(name, _THEMES["paper"]))
    th["_name"] = name if name in _THEMES else "paper"
    return th


def agent_color(th, i):
    return th["agents"][i % len(th["agents"])]


def battery_color(th, frac):
    """Green above 50%, amber down to 20%, red below — the usual battery ramp."""
    if frac >= 0.5:
        return th["ok"]
    if frac >= 0.2:
        return th["warn"]
    return th["bad"]


def _mix(c1, c2, t):
    """Blend two hex colours, t=0 -> c1, t=1 -> c2."""
    import matplotlib.colors as mcolors

    a = np.array(mcolors.to_rgb(c1))
    b = np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex(a * (1.0 - t) + b * t)


# ======================================================================
# Procedural terrain
# ======================================================================
#
# The benchmarks are drone-swarm tasks over open ground, so the ground is drawn
# as ground: a low-frequency noise field mapped to a dust / dry-grass / sage
# ramp, with darker scrub patches and a scatter of dust speckles on top.
#
# Two properties matter more than realism here:
#   * it must be QUIET. Everything that carries meaning — drones, tethers, radio
#     links, battery pills — sits on top of it, so the terrain is generated at
#     low contrast and then blended toward the theme's flat ground colour.
#   * it must be STABLE. The same figure is redrawn every timestep, so the noise
#     is cached by (theme, seed, resolution); regenerating per frame would make
#     the ground crawl in a GIF.

# Per theme: (low ground, mid ground, high ground, scrub tint, dust speck colour)
_TERRAIN = {
    "paper": ("#CDBE97", "#DCD0AC", "#E9E1C6", "#8FA968", "#FFF6E2"),
    "night": ("#131B19", "#182320", "#1E2B26", "#26402F", "#9CB8A6"),
}

_TERRAIN_CACHE = {}


def _smooth_noise(h, w, res, rng):
    """One octave of value noise: a coarse random grid, smoothstep-interpolated
    up to (h, w). Plain numpy — no scipy dependency."""
    grid = rng.random((res + 1, res + 1))

    def axis(n, size):
        t = np.linspace(0, size, n, endpoint=False)
        i0 = np.floor(t).astype(int)
        f = t - i0
        return i0, np.minimum(i0 + 1, size), f * f * (3 - 2 * f)   # smoothstep

    y0, y1, ty = axis(h, res)
    x0, x1, tx = axis(w, res)
    top = grid[np.ix_(y0, x0)] * (1 - tx) + grid[np.ix_(y0, x1)] * tx
    bot = grid[np.ix_(y1, x0)] * (1 - tx) + grid[np.ix_(y1, x1)] * tx
    return top * (1 - ty[:, None]) + bot * ty[:, None]


def _fractal_noise(h, w, seed, octaves=5, res=3):
    """Sum of octaves, normalised to [0, 1]."""
    rng = np.random.default_rng(seed)
    out, amp, total = np.zeros((h, w)), 1.0, 0.0
    for o in range(octaves):
        out += amp * _smooth_noise(h, w, res * (2 ** o), rng)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / max(1e-9, float(np.ptp(out)))


def terrain_image(th, theme_name="paper", seed=7, size=640, contrast=0.72):
    """Ground texture for `theme_name`, cached. Returns (rgb, veg).

    `rgb` is (size, size, 3); `veg` is the (size, size) vegetation mask, reused
    by `_scrub` so bushes land on the green patches instead of at random.

    `contrast` in [0, 1] scales how far the texture departs from the theme's
    flat ground colour. The texture is deliberately FINE-grained rather than
    high-contrast: a few big soft blobs read as clouds, whereas fine variation
    at moderate contrast reads as ground and stays behind the drones.
    """
    import matplotlib.colors as mcolors

    key = (theme_name, seed, size, round(contrast, 3))
    if key in _TERRAIN_CACHE:
        return _TERRAIN_CACHE[key]

    low, mid, high, scrub, _ = _TERRAIN.get(theme_name, _TERRAIN["paper"])
    ground  = _fractal_noise(size, size, seed, octaves=6, res=6)
    patches = _fractal_noise(size, size, seed + 991, octaves=5, res=7)
    grain   = _fractal_noise(size, size, seed + 17, octaves=2, res=48)

    ramp = np.array([mcolors.to_rgb(c) for c in (low, mid, high)])
    t = np.clip(ground, 0, 1) * 2.0
    i = np.clip(t.astype(int), 0, 1)
    f = (t - i)[..., None]
    img = ramp[i] * (1 - f) + ramp[i + 1] * f

    # scrub: green only where the patch field is high, with a soft shoulder
    veg = np.clip((patches - 0.55) / 0.30, 0.0, 1.0)
    img = (img * (1 - veg[..., None] * 0.58)
           + np.array(mcolors.to_rgb(scrub)) * (veg[..., None] * 0.58))

    # fine grain, so the ground still has texture at 300 dpi
    img *= (0.955 + 0.09 * grain)[..., None]

    # pull the whole thing back toward the flat theme ground
    flat = np.array(mcolors.to_rgb(th["field_top"]))
    img = flat * (1 - contrast) + img * contrast

    # float32: the cache holds one of these per (env, theme), and float64 at
    # 640x640x3 is ~10 MB apiece for no visible benefit.
    out = (np.clip(img, 0, 1).astype(np.float32), veg.astype(np.float32))
    _TERRAIN_CACHE[key] = out
    return out


def _dust(ax, x0, x1, y0, y1, th, theme_name, seed=7, n=260, zorder=0.45):
    """A light scatter of grit over the terrain, so the ground has grain at
    print resolution rather than looking like a blurred gradient."""
    rng = np.random.default_rng(seed + 31)
    color = _TERRAIN.get(theme_name, _TERRAIN["paper"])[4]
    xs = rng.uniform(x0, x1, n)
    ys = rng.uniform(y0, y1, n)
    ax.scatter(xs, ys, s=rng.uniform(0.4, 3.2, n), c=color,
               alpha=0.30, linewidths=0, zorder=zorder)


def _scrub(ax, x0, x1, y0, y1, th, theme_name, veg, seed=7, n=620,
           row_range=None, zorder=0.5):
    """Sparse bushes, placed where the vegetation mask is high.

    Seen from above, scrub is what makes ground read as ground: a smooth green
    wash on its own looks like a stain, while discrete clumps sitting inside the
    green patches look like a field.
    """
    rng = np.random.default_rng(seed + 77)
    scrub_color = _TERRAIN.get(theme_name, _TERRAIN["paper"])[3]
    dark = _mix(scrub_color, "#000000", 0.30)
    h, w = veg.shape
    r0, r1 = row_range if row_range else (0, h)

    u, v = rng.random(n), rng.random(n)
    rows = (r0 + v * (r1 - r0)).astype(int).clip(0, h - 1)
    cols = (u * (w - 1)).astype(int)
    keep = veg[rows, cols] > (0.35 + 0.45 * rng.random(n))
    if not keep.any():
        return
    xs = x0 + u[keep] * (x1 - x0)
    ys = y0 + v[keep] * (y1 - y0)
    sizes = rng.uniform(2.0, 11.0, int(keep.sum()))
    ax.scatter(xs, ys, s=sizes, c=dark, alpha=0.26, linewidths=0, zorder=zorder)
    ax.scatter(xs, ys, s=sizes * 0.35, c=scrub_color, alpha=0.34, linewidths=0,
               zorder=zorder + 0.01)


# ======================================================================
# Canvas / stage
# ======================================================================

_INTERACTIVE_READY = False


def stage(figsize=(8.6, 9.0), theme_name="paper", save_dir=None, window_title=None):
    """Create (or reuse) the figure and return (fig, ax, th).

    When `save_dir` is set the Agg backend is forced, so frames render
    identically on a headless machine; otherwise an interactive window is used.
    """
    global _INTERACTIVE_READY
    import matplotlib

    if save_dir is not None and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    th = theme(theme_name)
    if save_dir is None and not _INTERACTIVE_READY:
        plt.ion()
        _INTERACTIVE_READY = True

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(th["bg"])
    if window_title is not None:
        try:
            fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass
    return fig, ax, th


def stage_split(figsize=(11.0, 8.6), theme_name="paper", save_dir=None,
                window_title=None, width_ratios=(1.0, 0.44)):
    """Like `stage`, but returns (fig, ax_field, ax_panel, th).

    The right-hand axes is a telemetry column — per-drone battery/status rows
    that a purely spatial frame cannot carry.
    """
    global _INTERACTIVE_READY
    import matplotlib

    if save_dir is not None and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    th = theme(theme_name)
    if save_dir is None and not _INTERACTIVE_READY:
        plt.ion()
        _INTERACTIVE_READY = True

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios), wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor(th["bg"])
    if window_title is not None:
        try:
            fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass
    return fig, ax, axp, th


def reuse(fig, ax, theme_name="paper", panel=None):
    """Clear an existing figure for the next frame and return its theme."""
    th = theme(theme_name)
    ax.clear()
    if panel is not None:
        panel.clear()
    fig.patch.set_facecolor(th["bg"])
    return th


def panel(axp, th, title=None):
    """Prepare the telemetry column: unit coordinates, framed card, optional
    title. Returns nothing; draw with `panel_row` / raw axes calls after."""
    from matplotlib.patches import FancyBboxPatch

    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1)
    axp.set_xticks([])
    axp.set_yticks([])
    axp.set_facecolor(th["bg"])
    for spine in axp.spines.values():
        spine.set_visible(False)
    axp.add_patch(FancyBboxPatch(
        (0.015, 0.005), 0.97, 0.99,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        facecolor=th["panel"], edgecolor=th["panel_edge"], lw=1.2, alpha=0.75,
        zorder=0, transform=axp.transAxes,
    ))
    if title:
        axp.text(0.055, 0.965, title.upper(), ha="left", va="center",
                 color=th["muted"], fontsize=7.4, fontweight="bold", zorder=2)


def field(ax, x0, x1, y0, y1, th, grid_step=1, major_every=5, border=True,
          pad=0.62, terrain=True, terrain_seed=7, terrain_contrast=0.82):
    """Ground + survey grid. Call first; everything else layers on.

    x0/x1/y0/y1 are the DATA bounds of the playing field (inclusive), and the
    axes limits are padded slightly around them. `terrain=False` falls back to
    the plain two-stop gradient (what the blueprint theme wants, since a
    schematic should not pretend to be a photograph).
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_facecolor(th["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    theme_name = th.get("_name", "paper")
    extent = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
    if terrain and theme_name in _TERRAIN:
        img, veg = terrain_image(th, theme_name, seed=terrain_seed,
                                 contrast=terrain_contrast)
        ax.imshow(img, extent=extent, origin="lower", interpolation="bilinear",
                  aspect="auto", zorder=0)
        _scrub(ax, x0 - pad, x1 + pad, y0 - pad, y1 + pad, th, theme_name, veg,
               seed=terrain_seed)
        _dust(ax, x0 - pad, x1 + pad, y0 - pad, y1 + pad, th, theme_name,
              seed=terrain_seed)
    else:
        cmap = LinearSegmentedColormap.from_list(
            "field", [th["field_bottom"], th["field_top"]]
        )
        ramp = np.linspace(0.0, 1.0, 256)[:, None] * np.ones((1, 8))
        ax.imshow(ramp, extent=extent, cmap=cmap, origin="lower",
                  interpolation="bicubic", aspect="auto", zorder=0)
    # AFTER imshow: `aspect="auto"` above sets the axes aspect, which would
    # otherwise squash every drone glyph into an ellipse on a non-square axes.
    ax.set_aspect("equal", adjustable="box")

    if grid_step:
        on_terrain = terrain and theme_name in _TERRAIN
        # Over terrain the pale beige grid disappears, so switch to a white
        # survey overlay; on a flat background keep the original muted lines.
        c_min, c_maj = ((th["halo"], th["halo"]) if on_terrain
                        else (th["grid"], th["grid_major"]))
        a_min, a_maj = (0.34, 0.62) if on_terrain else (0.55, 0.85)
        xs = np.arange(math.ceil(x0), math.floor(x1) + 1, grid_step)
        ys = np.arange(math.ceil(y0), math.floor(y1) + 1, grid_step)
        for x in xs:
            major = (int(x) % major_every == 0)
            ax.plot([x, x], [y0 - 0.30, y1 + 0.30],
                    color=c_maj if major else c_min,
                    lw=1.0 if major else 0.6, zorder=0.8,
                    alpha=a_maj if major else a_min)
        for y in ys:
            major = (int(y) % major_every == 0)
            ax.plot([x0 - 0.30, x1 + 0.30], [y, y],
                    color=c_maj if major else c_min,
                    lw=1.0 if major else 0.6, zorder=0.8,
                    alpha=a_maj if major else a_min)

    if border:
        ax.add_patch(Rectangle(
            (x0 - pad * 0.72, y0 - pad * 0.72),
            (x1 - x0) + 1.44 * pad, (y1 - y0) + 1.44 * pad,
            facecolor="none", edgecolor=th["grid_major"], lw=1.3, alpha=0.9,
            zorder=0.9,
        ))


def plain_stage_bg(ax, x0, x1, y0, y1, th, pad=0.62, terrain_seed=7,
                   terrain=True, terrain_contrast=0.72):
    """Terrain with NO survey grid — for the abstract (gridless) benchmarks."""
    field(ax, x0, x1, y0, y1, th, grid_step=0, pad=pad, terrain=terrain,
          terrain_seed=terrain_seed, terrain_contrast=terrain_contrast)


def sky_field(ax, x0, x1, y0, y1, th, horizon, pad=0.25, terrain_seed=7,
              terrain=True, terrain_contrast=0.85):
    """Ground below `horizon`, hazy sky above it.

    For the scenes drawn in elevation rather than plan (drones hovering ABOVE a
    crate on the ground): a top-down terrain filling the whole panel would put
    dirt in the sky.
    """
    from matplotlib.colors import LinearSegmentedColormap

    theme_name = th.get("_name", "paper")
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_facecolor(th["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # sky: a soft vertical haze, lighter toward the horizon
    sky_top = _mix(th["field_top"], th["comm"], 0.16 if theme_name == "paper" else 0.10)
    sky_low = _mix(th["field_top"], "#FFFFFF" if theme_name == "paper" else "#2A3644", 0.45)
    cmap = LinearSegmentedColormap.from_list("sky", [sky_low, sky_top])
    ramp = np.linspace(0.0, 1.0, 256)[:, None] * np.ones((1, 8))
    ax.imshow(ramp, extent=(x0 - pad, x1 + pad, horizon, y1 + pad), cmap=cmap,
              origin="lower", interpolation="bicubic", aspect="auto", zorder=0)

    if terrain and theme_name in _TERRAIN:
        # A shallow slice of the terrain field, stretched: seen in elevation the
        # ground is foreshortened, so a square patch would read as a wall.
        img, veg = terrain_image(th, theme_name, seed=terrain_seed,
                                 contrast=terrain_contrast)
        band = img.shape[0] // 4
        ax.imshow(img[:band], origin="lower",
                  extent=(x0 - pad, x1 + pad, y0 - pad, horizon),
                  interpolation="bilinear", aspect="auto", zorder=0)
        _scrub(ax, x0 - pad, x1 + pad, y0 - pad, horizon, th, theme_name, veg,
               seed=terrain_seed, n=320, row_range=(0, band))
        _dust(ax, x0 - pad, x1 + pad, y0 - pad, horizon, th, theme_name,
              seed=terrain_seed, n=120)
    ax.plot([x0 - pad, x1 + pad], [horizon, horizon], color=th["grid_major"],
            lw=1.2, alpha=0.75, zorder=0.6)
    ax.set_aspect("equal", adjustable="box")


def compass(ax, x, y, th, size=0.55, label="N"):
    """Small north arrow, so the frame reads as a field survey."""
    ax.annotate(
        "", xy=(x, y + size), xytext=(x, y - size * 0.35),
        arrowprops=dict(arrowstyle="-|>", color=th["muted"], lw=1.3, alpha=0.8),
        annotation_clip=False, zorder=5,
    )
    ax.text(x, y + size * 1.12, label, ha="center", va="bottom", color=th["muted"],
            fontsize=7.2, fontweight="bold", zorder=5)


# ======================================================================
# Glyphs
# ======================================================================

_ROTOR_OFFSETS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def drone(
    ax,
    x,
    y,
    size=0.42,
    color="#2F6F9F",
    th=None,
    alpha=1.0,
    label=None,
    label_color=None,
    battery=None,
    ring_color=None,
    ring_lw=2.4,
    halo=None,
    halo_alpha=0.22,
    carrying=False,
    dimmed=False,
    zorder=6,
    shadow=True,
):
    """A top-down quadcopter: soft shadow, four blurred rotors, body, canopy.

    battery : None, or a float in [0, 1] drawn as an arc gauge around the drone.
    halo    : colour of a soft disc behind the drone (used for "committed",
              "charging", "in-mesh" emphasis).
    label   : short text drawn on the body (e.g. "W3", "S").
    """
    from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Wedge

    th = th or theme()
    a = alpha * (0.42 if dimmed else 1.0)
    edge = _mix(color, "#000000", 0.42)
    arm = size * 0.78

    if halo is not None:
        ax.add_patch(Circle((x, y), size * 1.62, facecolor=halo, edgecolor="none",
                            alpha=halo_alpha * alpha, zorder=zorder - 1.5))
        ax.add_patch(Circle((x, y), size * 1.62, facecolor="none", edgecolor=halo,
                            lw=1.1, alpha=0.55 * alpha, zorder=zorder - 1.4))

    if shadow:
        ax.add_patch(Ellipse((x + 0.06 * size, y - 0.34 * size), 2.05 * size, 0.80 * size,
                             facecolor=th["shadow"], edgecolor="none",
                             alpha=0.13 * a, zorder=zorder - 1.2))

    for dx, dy in _ROTOR_OFFSETS:
        cx, cy = x + dx * arm, y + dy * arm
        ax.plot([x, cx], [y, cy], color=edge, lw=2.1, solid_capstyle="round",
                alpha=a, zorder=zorder)
        # rotor disc (motion blur) + rim
        ax.add_patch(Circle((cx, cy), size * 0.50, facecolor=color, edgecolor="none",
                            alpha=0.16 * a, zorder=zorder + 0.1))
        ax.add_patch(Circle((cx, cy), size * 0.36, facecolor="none", edgecolor=edge,
                            lw=1.15, alpha=0.85 * a, zorder=zorder + 0.2))
        ax.add_patch(Circle((cx, cy), size * 0.09, facecolor=edge, edgecolor="none",
                            alpha=a, zorder=zorder + 0.25))

    body_w, body_h = size * 0.95, size * 0.80
    ax.add_patch(FancyBboxPatch(
        (x - body_w / 2, y - body_h / 2), body_w, body_h,
        boxstyle="round,pad=0.0,rounding_size=%.3f" % (size * 0.22),
        facecolor=color, edgecolor=edge, lw=1.3, alpha=a, zorder=zorder + 0.4,
    ))
    # canopy highlight
    ax.add_patch(Ellipse((x, y + size * 0.14), body_w * 0.52, body_h * 0.30,
                         facecolor=th["halo"], edgecolor="none",
                         alpha=0.22 * a, zorder=zorder + 0.5))

    if carrying:
        # slung cargo underneath, on two lines
        cw = size * 0.72
        for sx in (-1, 1):
            ax.plot([x + sx * body_w * 0.28, x + sx * cw * 0.42],
                    [y - body_h * 0.42, y - size * 1.02],
                    color=edge, lw=1.0, alpha=a, zorder=zorder + 0.3)
        ax.add_patch(FancyBboxPatch(
            (x - cw / 2, y - size * 1.36), cw, size * 0.46,
            boxstyle="round,pad=0.0,rounding_size=%.3f" % (size * 0.07),
            facecolor=th["crate"], edgecolor=th["crate_edge"], lw=1.0,
            alpha=a, zorder=zorder + 0.35,
        ))

    if battery is not None:
        frac = float(np.clip(battery, 0.0, 1.0))
        r = size * 1.30
        ax.add_patch(Wedge((x, y), r, 0, 360, width=size * 0.15,
                           facecolor=th["muted"], edgecolor="none",
                           alpha=0.30 * a, zorder=zorder + 0.6))
        if frac > 0.001:
            bc = ring_color or battery_color(th, frac)
            ax.add_patch(Wedge((x, y), r, 90.0, 90.0 + 360.0 * frac,
                               width=size * 0.15, facecolor=bc, edgecolor="none",
                               alpha=0.95 * a, zorder=zorder + 0.7))

    if label:
        ax.text(x, y - size * 0.02, str(label), ha="center", va="center",
                color=label_color or th["halo"], fontsize=7.4, fontweight="bold",
                alpha=min(1.0, a + 0.15), zorder=zorder + 0.8)

    if ring_color is not None and battery is None:
        ax.add_patch(Circle((x, y), size * 1.30, facecolor="none",
                            edgecolor=ring_color, lw=ring_lw, alpha=0.9 * a,
                            zorder=zorder + 0.6))


def crate(ax, x, y, size=0.92, th=None, label=None, sub=None, state="active",
          zorder=3):
    """Cargo box: corner brackets + strap bands + a requirement placard.

    state: "active" (full colour), "pending" (muted), "done" (faded + check).
    """
    from matplotlib.patches import FancyBboxPatch, Ellipse

    th = th or theme()
    alpha = {"active": 1.0, "lifting": 1.0, "pending": 0.55, "done": 0.35}.get(state, 1.0)
    face = th["crate"] if state != "done" else _mix(th["crate"], th["bg"], 0.55)
    edge = th["crate_edge"]
    hs = size / 2.0

    ax.add_patch(Ellipse((x + 0.03 * size, y - 0.58 * size), 1.28 * size, 0.42 * size,
                         facecolor=th["shadow"], edgecolor="none", alpha=0.14 * alpha,
                         zorder=zorder - 0.5))
    ax.add_patch(FancyBboxPatch(
        (x - hs, y - hs), size, size,
        boxstyle="round,pad=0.0,rounding_size=%.3f" % (size * 0.10),
        facecolor=face, edgecolor=edge, lw=1.6, alpha=alpha, zorder=zorder,
    ))
    # strap bands
    for f in (-0.22, 0.22):
        ax.plot([x - hs * 0.98, x + hs * 0.98], [y + f * size, y + f * size],
                color=edge, lw=1.5, alpha=0.55 * alpha, zorder=zorder + 0.1)
    ax.plot([x, x], [y - hs * 0.98, y + hs * 0.98],
            color=edge, lw=1.5, alpha=0.45 * alpha, zorder=zorder + 0.1)
    # corner brackets
    b = size * 0.26
    for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        cx, cy = x + sx * hs, y + sy * hs
        ax.plot([cx, cx - sx * b], [cy, cy], color=edge, lw=2.4, alpha=0.9 * alpha,
                solid_capstyle="round", zorder=zorder + 0.2)
        ax.plot([cx, cx], [cy, cy - sy * b], color=edge, lw=2.4, alpha=0.9 * alpha,
                solid_capstyle="round", zorder=zorder + 0.2)

    if label:
        ax.text(x, y + size * 0.02, str(label), ha="center", va="center",
                color="#FFFFFF" if state != "done" else th["muted"],
                fontsize=11.5, fontweight="bold", zorder=zorder + 0.4,
                path_effects=_stroke(edge, 2.2))
    if sub:
        ax.text(x, y - size * 1.25, str(sub), ha="center", va="top",
                color=th["muted"], fontsize=7.2, fontweight="bold",
                zorder=zorder + 0.4, path_effects=_stroke(th["field_top"], 2.6))
    if state in ("done", "lifting"):
        # A just-lifted crate is drawn ABOVE the lift tethers (which sit at
        # zorder ~5.5), so the outcome stays readable in the busiest frame.
        top = (zorder + 3.5) if state == "lifting" else (zorder + 0.5)
        ax.text(x, y, "✓", ha="center", va="center", color=th["ok"],
                fontsize=22 if state == "lifting" else 20, fontweight="bold",
                alpha=0.95, zorder=top,
                path_effects=_stroke("#FFFFFF", 2.6) if state == "lifting" else None)
        if state == "lifting":
            from matplotlib.patches import Rectangle

            ax.add_patch(Rectangle((x - hs * 1.32, y - hs * 1.32), size * 1.32,
                                   size * 1.32, facecolor="none",
                                   edgecolor=th["ok"], lw=2.0, alpha=0.85,
                                   zorder=top - 0.2))


def station(ax, x, y, size=1.5, n_ports=2, port_states=None, th=None, zorder=2,
            label="CHARGING STATION"):
    """Charging pad: hatched apron, a port bay per capacity slot, ⚡ mast.

    port_states: list of length n_ports, each "free" | "busy".
    """
    from matplotlib.patches import Circle, FancyBboxPatch

    th = th or theme()
    port_states = port_states or ["free"] * n_ports
    hs = size / 2.0

    ax.add_patch(Circle((x, y), size * 0.72, facecolor=th["pad"], edgecolor="none",
                        alpha=0.10, zorder=zorder - 0.6))
    ax.add_patch(FancyBboxPatch(
        (x - hs, y - hs), size, size,
        boxstyle="round,pad=0.0,rounding_size=%.3f" % (size * 0.16),
        facecolor=th["pad"], edgecolor=th["pad_edge"], lw=1.8, alpha=0.92,
        zorder=zorder,
    ))
    # landing "H"
    ax.text(x, y + size * 0.04, "⌁", ha="center", va="center", color=th["halo"],
            fontsize=15, fontweight="bold", alpha=0.30, zorder=zorder + 0.1)

    # port slots along the bottom edge
    pw = size * 0.72 / max(1, n_ports)
    for p in range(n_ports):
        px = x - size * 0.36 + pw * (p + 0.5)
        busy = port_states[p] == "busy"
        ax.add_patch(FancyBboxPatch(
            (px - pw * 0.36, y - hs * 0.86), pw * 0.72, size * 0.17,
            boxstyle="round,pad=0.0,rounding_size=%.3f" % (size * 0.03),
            facecolor=th["ok"] if busy else th["panel"],
            edgecolor=th["pad_edge"], lw=1.0, alpha=0.95, zorder=zorder + 0.2,
        ))
    if label:
        ax.text(x, y - hs - size * 0.20, label, ha="center", va="top",
                color=th["muted"], fontsize=6.8, fontweight="bold",
                zorder=zorder + 0.3)


def bolt(ax, x, y, size=0.30, color=None, th=None, alpha=1.0, zorder=9):
    """Charging bolt glyph."""
    th = th or theme()
    ax.text(x, y, "⚡", ha="center", va="center", fontsize=size * 46,
            color=color or th["warn"], alpha=alpha, zorder=zorder,
            path_effects=_stroke(th["halo"], 2.0))


def comm_link(ax, p, q, th=None, color=None, lw=1.5, alpha=0.85, style=(0, (4, 3)),
              glow=True, zorder=4):
    """Dashed radio link with a soft glow underneath."""
    th = th or theme()
    c = color or th["comm"]
    if glow:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw * 3.4, alpha=0.11 * alpha,
                solid_capstyle="round", zorder=zorder - 0.1)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, alpha=alpha,
            linestyle=style, dash_capstyle="round", zorder=zorder)


def comm_disk(ax, cx, cy, radius, th=None, color=None, alpha=0.09, lw=1.4,
              manhattan=True, zorder=1.2, label=None, edge_alpha=0.75):
    """Radio footprint. Manhattan radius draws the L1 diamond the envs use.

    `edge_alpha` is separate from the fill so the same call can draw a loud
    radio disk or a barely-there reference ring (e.g. a cohort spawn ring).
    """
    from matplotlib.patches import Circle, Polygon

    th = th or theme()
    c = color or th["comm"]
    if radius <= 0:
        radius = 0.46  # a zero-radius gate still deserves a visible footprint
        manhattan = False
    if manhattan:
        pts = [(cx + radius, cy), (cx, cy + radius), (cx - radius, cy), (cx, cy - radius)]
        shape = lambda **kw: Polygon(pts, closed=True, **kw)
    else:
        shape = lambda **kw: Circle((cx, cy), radius, **kw)
    if alpha > 0:
        ax.add_patch(shape(facecolor=c, edgecolor="none", alpha=alpha, zorder=zorder))
    ax.add_patch(shape(facecolor="none", edgecolor=c, lw=lw, alpha=edge_alpha,
                       linestyle=(0, (5, 3)), zorder=zorder + 0.1))
    if label:
        # below the disk: the hub drone sits at its centre, so a label above
        # would land on top of the glyph.
        ax.text(cx, cy - radius - 0.14, label, ha="center", va="top",
                color=c, fontsize=6.8, fontweight="bold", alpha=0.95,
                zorder=zorder + 0.2, path_effects=_stroke(th["field_top"], 2.4))


def trail(ax, points, color, th=None, alpha=0.55, lw=2.2, zorder=2.5, dots=True):
    """Fading breadcrumb path — makes a still frame show where a drone came from.

    Drawn as a dotted, fading polyline so it never reads as a stray grid line.
    """
    if points is None or len(points) < 2:
        return
    th = th or theme()
    n = len(points)
    for i in range(n - 1):
        f = (i + 1) / float(n)
        ax.plot(
            [points[i][0], points[i + 1][0]],
            [points[i][1], points[i + 1][1]],
            color=color, lw=lw * (0.30 + 0.70 * f), alpha=alpha * f * 0.9,
            solid_capstyle="round", linestyle=(0, (1, 1.8)), zorder=zorder,
        )
        if dots and i % 2 == 0:
            ax.plot([points[i][0]], [points[i][1]], marker="o", markersize=1.9,
                    color=color, alpha=alpha * f * 0.8, zorder=zorder)


def panel_row(axp, y, tag, tag_color, frac, status, status_color, th,
              target=None, note=None, dimmed=False):
    """One telemetry line: colour chip, drone tag, battery pill, status word."""
    from matplotlib.patches import Circle

    a = 0.45 if dimmed else 1.0
    axp.add_patch(Circle((0.075, y), 0.0115, facecolor=tag_color,
                         edgecolor="none", alpha=a, zorder=3))
    axp.text(0.115, y, tag, ha="left", va="center", color=th["ink"],
             fontsize=8.2, fontweight="bold", alpha=a, zorder=3)
    battery_bar(axp, 0.255, y - 0.0125, 0.36, 0.025, frac, th, target=target,
                zorder=3)
    axp.text(0.645, y, f"{frac * 100:3.0f}%", ha="left", va="center",
             color=th["muted"], fontsize=7.2, alpha=a, zorder=3)
    axp.text(0.985, y, status, ha="right", va="center", color=status_color,
             fontsize=7.6, fontweight="bold", alpha=a, zorder=3)
    if note:
        axp.text(0.115, y - 0.030, note, ha="left", va="center",
                 color=th["muted"], fontsize=6.4, alpha=a, zorder=3)


def battery_bar(ax, x, y, w, h, frac, th=None, target=None, zorder=8, label=None):
    """Horizontal battery pill with an optional target tick."""
    from matplotlib.patches import FancyBboxPatch

    th = th or theme()
    frac = float(np.clip(frac, 0.0, 1.0))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=%.3f" % (h * 0.45),
        facecolor=th["panel"], edgecolor=th["panel_edge"], lw=0.9, alpha=0.95,
        zorder=zorder,
    ))
    if frac > 0.01:
        ax.add_patch(FancyBboxPatch(
            (x + h * 0.10, y + h * 0.16), max(h * 0.30, (w - h * 0.20) * frac), h * 0.68,
            boxstyle="round,pad=0.0,rounding_size=%.3f" % (h * 0.30),
            facecolor=battery_color(th, frac), edgecolor="none", alpha=0.95,
            zorder=zorder + 0.1,
        ))
    else:
        # A flat battery is the most important state in the charging envs, and an
        # empty white pill reads as "no data": outline it in the alarm colour.
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=%.3f" % (h * 0.45),
            facecolor="none", edgecolor=th["bad"], lw=1.6, alpha=0.95,
            zorder=zorder + 0.15,
        ))
    if target is not None:
        tx = x + h * 0.10 + (w - h * 0.20) * float(np.clip(target, 0.0, 1.0))
        ax.plot([tx, tx], [y - h * 0.16, y + h * 1.16], color=th["ink"], lw=1.1,
                alpha=0.65, zorder=zorder + 0.2)
    if label:
        ax.text(x + w + h * 0.35, y + h * 0.5, label, ha="left", va="center",
                color=th["muted"], fontsize=6.6, fontweight="bold", zorder=zorder + 0.2)


def tether(ax, p, q, taut=False, color=None, th=None, sag=0.55, lw=None,
           alpha=None, zorder=4.5):
    """Lift line from a drone (p) down to the cargo (q).

    A slack line sags and is drawn faint; a taut line is straight, solid and
    saturated. That contrast is the whole point: which drones actually
    committed this step is readable at a glance.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    th = th or theme()
    c = color or (th["ok"] if taut else th["muted"])
    lw = lw if lw is not None else (2.2 if taut else 1.1)
    alpha = alpha if alpha is not None else (0.92 if taut else 0.35)

    if taut:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=zorder)
        return
    mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0 - sag)
    path = Path([p, mid, q], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=c, lw=lw,
                           alpha=alpha, linestyle=(0, (3, 2.5)), zorder=zorder))


def badge_dot(ax, x, y, color, th=None, r=0.085, zorder=9):
    """Small status dot pinned to a drone — a compact stand-in for a text tag
    when several drones share a cell and labels would collide."""
    from matplotlib.patches import Circle

    th = th or theme()
    ax.add_patch(Circle((x, y), r * 1.5, facecolor=th["halo"], edgecolor="none",
                        alpha=0.85, zorder=zorder))
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none",
                        alpha=0.95, zorder=zorder + 0.1))


def _stroke(color, lw):
    """Text outline so labels stay readable over any background."""
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=lw, foreground=color)]


def stroke(color, lw=2.2):
    return _stroke(color, lw)


# ======================================================================
# HUD
# ======================================================================

# Figure-fraction layout. Everything the HUD draws hangs off these, so the
# blocks never collide when a caller adds or drops a row.
LAYOUT = dict(
    title_y=0.980,
    subtitle_y=0.955,
    stat_label_y=0.921,
    stat_value_y=0.899,
    axes_top=0.868,
    axes_bottom=0.108,
    timeline_y=0.076,
    legend_y=0.046,
    footer_y=0.018,
    left=0.048,
    right=0.958,
)


def header(fig, ax, title, subtitle=None, th=None):
    """Title block above the field."""
    th = th or theme()
    ax.set_title("")
    fig.text(LAYOUT["left"], LAYOUT["title_y"], title, ha="left", va="top",
             color=th["ink"], fontsize=14.0, fontweight="bold")
    if subtitle:
        fig.text(LAYOUT["left"], LAYOUT["subtitle_y"], subtitle, ha="left",
                 va="top", color=th["muted"], fontsize=8.3)


def stat_row(fig, stats, th=None, x0=None, x1=None):
    """A row of key/value chips: stats = [(label, value, colour_or_None), ...]."""
    th = th or theme()
    if not stats:
        return
    x0 = LAYOUT["left"] if x0 is None else x0
    x1 = LAYOUT["right"] if x1 is None else x1
    n = len(stats)
    span = (x1 - x0) / n
    for i, item in enumerate(stats):
        label, value = item[0], item[1]
        color = item[2] if len(item) > 2 and item[2] else th["ink"]
        x = x0 + span * i
        fig.text(x, LAYOUT["stat_label_y"], str(label).upper(), ha="left",
                 va="center", color=th["muted"], fontsize=6.5, fontweight="bold")
        fig.text(x, LAYOUT["stat_value_y"], str(value), ha="left", va="center",
                 color=color, fontsize=11.5, fontweight="bold")


def status_pill(fig, text, color, th=None, x=None, y=None, fontsize=9.5):
    """Right-aligned outcome badge (CORRECT LIFT / BROWNOUT / ...)."""
    th = th or theme()
    fig.text(LAYOUT["right"] if x is None else x,
             LAYOUT["title_y"] if y is None else y,
             f"  {text}  ", ha="right", va="top", color="#FFFFFF",
             fontsize=fontsize, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.45", facecolor=color,
                       edgecolor="none", alpha=0.95))


def legend_row(fig, items, th=None, y=None, x0=None):
    """Bottom legend: items = [(marker_color, text), ...]."""
    th = th or theme()
    x = LAYOUT["left"] if x0 is None else x0
    y = LAYOUT["legend_y"] if y is None else y
    for color, text in items:
        fig.text(x, y, "●", ha="left", va="center", color=color, fontsize=9.5)
        fig.text(x + 0.015, y, text, ha="left", va="center", color=th["muted"],
                 fontsize=7.4)
        x += 0.021 + 0.0064 * len(text)


def footer(fig, text, th=None, y=None, x=None, right=None):
    """Bottom caption. `right` is where the policy label goes — it can be long
    (a checkpoint name), and it would overflow the header subtitle."""
    th = th or theme()
    y = LAYOUT["footer_y"] if y is None else y
    fig.text(LAYOUT["left"] if x is None else x, y, text, ha="left", va="center",
             color=th["muted"], fontsize=7.0)
    if right:
        fig.text(LAYOUT["right"], y, str(right), ha="right", va="center",
                 color=th["muted"], fontsize=7.0)


def timeline(fig, t, t_max, th=None, rect=None):
    """Thin episode-progress bar under the field."""
    th = th or theme()
    if rect is None:
        rect = (LAYOUT["left"], LAYOUT["timeline_y"],
                LAYOUT["right"] - LAYOUT["left"], 0.006)
    x, y, w, h = rect
    frac = 0.0 if not t_max else min(1.0, float(t) / float(t_max))
    fig.patches.append(_fig_rect(fig, x, y, w, h, th["panel_edge"], 0.55))
    if frac > 0:
        fig.patches.append(_fig_rect(fig, x, y, w * frac, h, th["ink"], 0.55))


def _fig_rect(fig, x, y, w, h, color, alpha):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=color,
                     edgecolor="none", alpha=alpha, zorder=5)


# ======================================================================
# Output
# ======================================================================

def finish(fig, save_dir=None, name=None, dpi=140, pause=0.35, tight=True):
    """Show the frame (interactive) or write it to `save_dir/name.png`.

    Returns the written path, or None when displayed interactively.
    """
    import matplotlib.pyplot as plt

    if tight:
        fig.subplots_adjust(left=LAYOUT["left"] - 0.008, right=LAYOUT["right"] + 0.008,
                            top=LAYOUT["axes_top"], bottom=LAYOUT["axes_bottom"])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name or 'frame'}.png")
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        return path

    plt.draw()
    plt.pause(max(0.001, pause))
    return None


def write_gif(frame_paths, out_path, fps=4, loop=0):
    """Stitch PNG frames into a GIF (imageio if present, else Pillow)."""
    frame_paths = [p for p in frame_paths if p and os.path.isfile(p)]
    if not frame_paths:
        raise ValueError("no frames to write")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(out_path, [imageio.imread(p) for p in frame_paths],
                        duration=1.0 / max(1, fps), loop=loop)
    except ImportError:
        from PIL import Image

        frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
                  for p in frame_paths]
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / max(1, fps)), loop=loop, optimize=True)
    return out_path


def write_mp4(frame_paths, out_path, fps=6):
    """Stitch frames into an MP4 (needs imageio-ffmpeg); returns None if absent."""
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(out_path, fps=fps)
        for p in frame_paths:
            writer.append_data(imageio.imread(p))
        writer.close()
        return out_path
    except Exception:
        return None


# ======================================================================
# Loader used by the env modules
# ======================================================================

def kit():
    """Return this module."""
    return _self()


def _self():
    import sys

    return sys.modules[__name__]
