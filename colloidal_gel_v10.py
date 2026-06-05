"""
=============================================================
 Brownian Dynamics — Coarsening & Arrest in Colloidal Gels
 v10 : version longue — ε_gel=8kBT, t_final=2000τ, N=500
      computed for multiple waiting times tw
=============================================================
 Changes from v7:
   - MSD tracking with unwrapped positions (no PBC folding)
   - Ageing protocol: MSD measured from multiple waiting times
     tw = [0, t_final/4, t_final/2, 3*t_final/4]
   - MSD(t-tw, tw): gel shows tw-dependence (ageing / non-ergodicity),
     liquid is tw-independent (ergodic)
   - New plots:  07_MSD_gel.png, 08_MSD_liquid.png,
                 09_MSD_ageing_comparison.png
   - Dashboard extended with MSD panel
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numba import njit

# ─────────────────────────────────────────────
#  PARAMETERS
# ─────────────────────────────────────────────
N          = 500
PHI        = 0.25
SIGMA      = 1.0
kBT        = 1.0
GAMMA      = 1.0

EPS_GEL    = 8.0      # gel: deep attractive well → kinetic arrest
EPS_LIQ    = 1.5      # liquid reference: weak attraction, stays ergodic

DT         = 5e-5
N_EQUIL    = 8_000
N_STEPS    = 40_000_000  # t_final = 2000 τ
SNAP_EVERY = 4_000_000   # 11 snapshots (t=0..2000)

NK_SHELLS  = 40
NK_MAX     = 25
SEED       = 42

# ── Ageing / MSD parameters ───────────────────────────────
# Waiting times tw at which we "start the clock" for MSD
# expressed as fraction of N_STEPS
TW_FRACS   = [0.0, 0.25, 0.50, 0.75]   # tw = fraction × t_final
MSD_EVERY  = 200_000                     # record MSD every N steps

R_CUT_GEL  = 2.5
R_CUT_LIQ  = 2.5        # same cutoff, only ε differs
R_CUT_WCA  = 2**(1/6)   # still used for equilibration

FORCE_CAP  = 500.0

PLOT_DIR   = "plots"

# matplotlib style: white background, primary colors
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "black",
    "axes.labelcolor":  "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "text.color":       "black",
    "grid.color":       "#cccccc",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "axes.grid":        True,
    "legend.framealpha": 0.8,
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
})

# Primary colors (matplotlib C0/C1/C2 cycle + extras)
C_GEL  = "#1f77b4"   # blue   — gel
C_LIQ  = "#d62728"   # red    — liquid
C_TIME = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf", "#aec7e8"]  # time series palette

# ─────────────────────────────────────────────
#  NUMBA FORCES  (with force cap)
# ─────────────────────────────────────────────

@njit(cache=True)
def compute_forces_numba(pos, L, eps, r_cut):
    N_     = pos.shape[0]
    forces = np.zeros((N_, 2))
    U_tot  = 0.0
    sig2   = SIGMA**2
    r_cut2 = r_cut**2
    ir2c   = sig2 / r_cut2
    ir6c   = ir2c**3
    U_cut  = 4.0 * eps * (ir6c*ir6c - ir6c)

    for i in range(N_ - 1):
        for j in range(i + 1, N_):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            r2 = dx*dx + dy*dy
            if r2 >= r_cut2 or r2 < 0.25:
                continue
            ir2  = sig2 / r2
            ir6  = ir2 * ir2 * ir2
            ir12 = ir6 * ir6
            U_tot += 4.0 * eps * (ir12 - ir6) - U_cut
            fr = 24.0 * eps / r2 * (2.0*ir12 - ir6)

            r_inv = 1.0 / (r2**0.5)
            f_mag = abs(fr) * r2**0.5
            if f_mag > FORCE_CAP:
                fr = FORCE_CAP * r_inv * (1.0 if fr > 0 else -1.0) / r2**0.5

            fx = fr * dx; fy = fr * dy
            forces[i, 0] += fx;  forces[i, 1] += fy
            forces[j, 0] -= fx;  forces[j, 1] -= fy

    return forces, U_tot


# ─────────────────────────────────────────────
#  MSD — unwrapped positions (no PBC folding)
# ─────────────────────────────────────────────

@njit(cache=True)
def unwrap_positions(pos_new, pos_old, pos_unwrapped, L):
    """Apply minimum-image convention to track displacement without folding."""
    N_ = pos_new.shape[0]
    pos_out = np.empty_like(pos_unwrapped)
    for i in range(N_):
        for d in range(2):
            delta = pos_new[i, d] - pos_old[i, d]
            delta -= L * round(delta / L)
            pos_out[i, d] = pos_unwrapped[i, d] + delta
    return pos_out


def compute_msd(pos_unwrapped, ref_pos):
    """MSD = <|r(t) - r(tw)|^2> averaged over all particles."""
    dr = pos_unwrapped - ref_pos
    return float(np.mean(dr[:, 0]**2 + dr[:, 1]**2))


# ─────────────────────────────────────────────
#  INITIALISATION
# ─────────────────────────────────────────────

def init_positions(N, phi, sigma, seed=42):
    L = np.sqrt(N * np.pi * (sigma / 2)**2 / phi)
    rng = np.random.default_rng(seed)
    n_side = int(np.ceil(np.sqrt(N)))
    spacing = L / n_side
    xs = (np.arange(n_side) + 0.5) * spacing
    grid = np.array([[x, y] for y in xs for x in xs])[:N]
    pos = grid + rng.uniform(-0.02*spacing, 0.02*spacing, size=(N, 2))
    return pos % L, L


def equilibrate(pos, L, n_steps, seed):
    rng = np.random.default_rng(seed + 99)
    dt_eq = 2e-5
    print(f"  Equilibration ({n_steps} steps, dt={dt_eq}) ...", end="", flush=True)
    for step in range(n_steps):
        frac    = min(1.0, (step + 1) / (n_steps * 0.3))
        eps_now = 0.05 + 0.95 * frac
        forces, _ = compute_forces_numba(pos, L, eps_now, R_CUT_WCA)
        noise  = rng.standard_normal(pos.shape)
        amp    = np.sqrt(2.0 * kBT * dt_eq / GAMMA)
        pos    = (pos + (dt_eq / GAMMA) * forces + amp * noise) % L
    print(" OK")
    return pos


def em_step(pos, L, eps, r_cut, rng):
    forces, U = compute_forces_numba(pos, L, eps, r_cut)
    noise   = rng.standard_normal(pos.shape)
    amp     = np.sqrt(2.0 * kBT * DT / GAMMA)
    pos_new = (pos + (DT / GAMMA) * forces + amp * noise) % L
    return pos_new, U


# ─────────────────────────────────────────────
#  STRUCTURE FACTOR S(k)
# ─────────────────────────────────────────────

def structure_factor(pos, L, nk_shells=NK_SHELLS, nk_max=NK_MAX):
    k0 = 2.0 * np.pi / L
    n  = len(pos)
    nx_arr, ny_arr = np.meshgrid(np.arange(-nk_max, nk_max+1),
                                  np.arange(-nk_max, nk_max+1))
    nx_arr = nx_arr.ravel(); ny_arr = ny_arr.ravel()
    n2     = nx_arr**2 + ny_arr**2
    sel    = (n2 >= 1) & (n2 <= nk_max**2)
    nx_arr, ny_arr, n2 = nx_arr[sel], ny_arr[sel], n2[sel]

    kx     = nx_arr * k0; ky = ny_arr * k0
    k_mags = np.sqrt(n2) * k0

    phase  = kx[:, None] * pos[:, 0] + ky[:, None] * pos[:, 1]
    Sk_all = np.abs(np.sum(np.exp(1j * phase), axis=1))**2 / n

    bins      = np.linspace(k_mags.min()*0.99, k_mags.max()*1.01, nk_shells+1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    idx       = np.clip(np.searchsorted(bins, k_mags, side='right') - 1, 0, nk_shells-1)
    S_binned  = np.zeros(nk_shells); counts = np.zeros(nk_shells)
    np.add.at(S_binned, idx, Sk_all); np.add.at(counts, idx, 1)
    valid = counts > 0
    S_binned[valid] /= counts[valid]
    return k_centers[valid], S_binned[valid]


# ─────────────────────────────────────────────
#  SIMULATION
# ─────────────────────────────────────────────

def run_simulation(mode="gel", seed=SEED):
    rng = np.random.default_rng(seed)
    pos, L = init_positions(N, PHI, SIGMA, seed=seed)

    eps   = EPS_GEL  if mode == "gel" else EPS_LIQ
    r_cut = R_CUT_GEL if mode == "gel" else R_CUT_LIQ
    label = f"GEL (LJ ε={EPS_GEL}kBT)" if mode == "gel" \
            else f"LIQUID (LJ ε={EPS_LIQ}kBT)"

    print(f"\n{'='*60}\n  Mode: {label}")
    print(f"  N={N}, φ={PHI}, dt={DT}, t_final={N_STEPS*DT:.0f}τ, L={L:.2f}σ")

    print("  Compiling numba ...", end="", flush=True)
    _ = compute_forces_numba(pos[:10], L, 1.0, R_CUT_WCA)
    _dum = np.zeros((10, 2))
    _ = unwrap_positions(pos[:10], pos[:10], _dum, L)
    print(" OK")

    pos = equilibrate(pos, L, N_EQUIL, seed)

    # ── MSD / ageing setup ─────────────────────────────────
    # Convert tw fractions to step indices
    tw_steps = [int(f * N_STEPS) for f in TW_FRACS]
    tw_times  = [s * DT for s in tw_steps]

    # For each tw: store reference (unwrapped) positions + MSD trace
    msd_refs   = [None] * len(tw_steps)   # reference unwrapped pos at tw
    msd_traces = [[] for _ in tw_steps]   # list of (t-tw, MSD) per tw

    # Unwrapped positions (updated every step)
    pos_unwrapped = pos.copy()
    pos_prev      = pos.copy()           # wrapped, previous step

    snapshots, pos_snaps = [], []

    for step in range(N_STEPS + 1):
        if step % SNAP_EVERY == 0:
            t = step * DT
            k_vals, S_vals = structure_factor(pos, L)
            _, U = compute_forces_numba(pos, L, eps, r_cut)
            i_pk = np.argmax(S_vals)
            print(f"  t={t:6.1f}τ  U={U:8.1f}kBT  "
                  f"k*={k_vals[i_pk]:.4f}σ⁻¹  S(k*)={S_vals[i_pk]:.3f}")
            snapshots.append((t, k_vals.copy(), S_vals.copy()))
            if len(pos_snaps) < 4:
                pos_snaps.append((t, pos.copy()))

        # ── Set MSD reference at waiting times ─────────────
        for idx, tw_s in enumerate(tw_steps):
            if step == tw_s:
                msd_refs[idx] = pos_unwrapped.copy()

        # ── Record MSD for each active tw ──────────────────
        if step % MSD_EVERY == 0:
            for idx, tw_s in enumerate(tw_steps):
                if step >= tw_s and msd_refs[idx] is not None:
                    lag = (step - tw_s) * DT
                    msd_val = compute_msd(pos_unwrapped, msd_refs[idx])
                    msd_traces[idx].append((lag, msd_val))

        if step < N_STEPS:
            pos_new, _ = em_step(pos, L, eps, r_cut, rng)
            # Unwrap: track continuous displacement across PBC
            pos_unwrapped = unwrap_positions(pos_new, pos, pos_unwrapped, L)
            pos_prev = pos.copy()
            pos = pos_new

    # Convert MSD traces to numpy arrays
    msd_data = []
    for idx, trace in enumerate(msd_traces):
        arr = np.array(trace)   # shape (n_pts, 2): col0=lag, col1=MSD
        msd_data.append((tw_times[idx], arr))

    return snapshots, pos, L, pos_snaps, msd_data


# ─────────────────────────────────────────────
#  PLOTTING  — individual files
# ─────────────────────────────────────────────

def savefig(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── 1. Particle positions snapshots (gel) ──────────────────
def plot_positions_snapshots(pos_snaps, L, label="gel"):
    n = min(len(pos_snaps), 4)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4.2))
    if n == 1: axes = [axes]
    color = C_GEL if label == "gel" else C_LIQ
    title_prefix = "Gel" if label == "gel" else "Liquid"
    for ax, (t, pos) in zip(axes, pos_snaps[:n]):
        ax.scatter(pos[:,0], pos[:,1], s=8, color=color,
                   alpha=0.7, linewidths=0)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_title(f"{title_prefix} — t = {t:.0f} τ")
        ax.set_xlabel("x [σ]"); ax.set_ylabel("y [σ]")
    fig.suptitle(
        f"Particle positions over time — {title_prefix}\n"
        f"N={N}, φ={PHI}, ε={'gel: '+str(EPS_GEL) if label=='gel' else 'liq: '+str(EPS_LIQ)} kBT",
        fontsize=11
    )
    fig.tight_layout()
    savefig(fig, f"01_positions_{label}.png")


# ── 2. S(k) evolution — gel ────────────────────────────────
def plot_sk_evolution(snaps, label="gel"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ns = len(snaps)
    for i, (t, k, S) in enumerate(snaps):
        col = C_TIME[i % len(C_TIME)]
        lbl = f"t = {t:.0f} τ" if i in [0, ns//4, ns//2, 3*ns//4, ns-1] else "_"
        ax.plot(k, S, color=col, lw=1.5, alpha=0.85, label=lbl)
        i_pk = np.argmax(S)
        ax.plot(k[i_pk], S[i_pk], 'o', color=col, ms=5, zorder=5)
    eps_val = EPS_GEL if label == "gel" else EPS_LIQ
    mode_str = "Gel" if label == "gel" else "Liquid"
    ax.set_xlabel("k [σ⁻¹]")
    ax.set_ylabel("S(k)")
    ax.set_title(
        f"Structure factor S(k) — {mode_str} (ε = {eps_val} kBT)\n"
        f"Circles mark the peak k* at each snapshot"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, f"02_Sk_evolution_{label}.png")


# ── 3. S(k) final comparison ──────────────────────────────
def plot_sk_comparison(snaps_gel, snaps_liq):
    fig, ax = plt.subplots(figsize=(7, 5))
    tg, kg, Sg = snaps_gel[-1]
    tl, kl, Sl = snaps_liq[-1]
    ax.plot(kg, Sg, color=C_GEL, lw=2.2, label=f"Gel (ε={EPS_GEL}kBT) — t={tg:.0f}τ")
    ax.plot(kl, Sl, color=C_LIQ, lw=2.2, ls="--",
            label=f"Liquid (ε={EPS_LIQ}kBT) — t={tl:.0f}τ")
    # mark peaks
    ig = np.argmax(Sg); il = np.argmax(Sl)
    ax.plot(kg[ig], Sg[ig], 'o', color=C_GEL, ms=8, zorder=5)
    ax.plot(kl[il], Sl[il], 's', color=C_LIQ, ms=8, zorder=5)
    ax.set_xlabel("k [σ⁻¹]")
    ax.set_ylabel("S(k)")
    ax.set_title(
        f"Final structure factor: Gel vs Liquid\n"
        f"t_final = {tg:.0f}τ,  N={N},  φ={PHI}"
    )
    ax.legend()
    fig.tight_layout()
    savefig(fig, "03_Sk_final_comparison.png")


# ── 4. k*(t) ──────────────────────────────────────────────
def plot_kstar(snaps_gel, snaps_liq):
    fig, ax = plt.subplots(figsize=(7, 5))
    t_g  = [s[0] for s in snaps_gel]
    ks_g = [s[1][np.argmax(s[2])] for s in snaps_gel]
    t_l  = [s[0] for s in snaps_liq]
    ks_l = [s[1][np.argmax(s[2])] for s in snaps_liq]
    ax.plot(t_g, ks_g, 'o-', color=C_GEL, lw=2, ms=6,
            label=f"Gel (ε={EPS_GEL}kBT)")
    ax.plot(t_l, ks_l, 's--', color=C_LIQ, lw=2, ms=6,
            label=f"Liquid (ε={EPS_LIQ}kBT)")
    ax.set_xlabel("t [τ]")
    ax.set_ylabel("k* [σ⁻¹]")
    ax.set_title(
        "Peak wavevector k*(t)\n"
        "Gel arrest → k* plateaus; Liquid coarsening → k* decreases"
    )
    ax.legend()
    fig.tight_layout()
    savefig(fig, "04_kstar_vs_time.png")


# ── 5. S(k*, t) ────────────────────────────────────────────
def plot_sstar(snaps_gel, snaps_liq):
    fig, ax = plt.subplots(figsize=(7, 5))
    t_g  = [s[0] for s in snaps_gel]
    Ss_g = [np.max(s[2]) for s in snaps_gel]
    t_l  = [s[0] for s in snaps_liq]
    Ss_l = [np.max(s[2]) for s in snaps_liq]
    ax.plot(t_g, Ss_g, 'o-', color=C_GEL, lw=2, ms=6,
            label=f"Gel (ε={EPS_GEL}kBT)")
    ax.plot(t_l, Ss_l, 's--', color=C_LIQ, lw=2, ms=6,
            label=f"Liquid (ε={EPS_LIQ}kBT)")
    ax.set_xlabel("t [τ]")
    ax.set_ylabel("S(k*)")
    ax.set_title(
        "Peak structure factor S(k*, t)\n"
        "Growing S(k*) → coarsening;  Flat → kinetic arrest"
    )
    ax.legend()
    fig.tight_layout()
    savefig(fig, "05_Sstar_vs_time.png")


# ── 6. Final positions comparison ─────────────────────────
def plot_final_positions(pos_gel, pos_liq, L):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pos, color, title in [
        (axes[0], pos_gel, C_GEL, f"Gel — final positions (ε={EPS_GEL}kBT)"),
        (axes[1], pos_liq, C_LIQ, f"Liquid — final positions (ε={EPS_LIQ}kBT)"),
    ]:
        ax.scatter(pos[:,0], pos[:,1], s=8, color=color, alpha=0.7, linewidths=0)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [σ]"); ax.set_ylabel("y [σ]")
    fig.suptitle(
        f"Final particle configurations — Gel vs Liquid\n"
        f"N={N}, φ={PHI}, t_final={N_STEPS*DT:.0f}τ",
        fontsize=11
    )
    fig.tight_layout()
    savefig(fig, "06_final_positions_comparison.png")


# ── 7. MSD vs lag time, one mode ───────────────────────────
def plot_msd(msd_data, label="gel"):
    """
    MSD(Δt, tw) for multiple waiting times tw.
    Gel: curves shift right as tw increases → ageing signature.
    Liquid: curves collapse → ergodic.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    color  = C_GEL if label == "gel" else C_LIQ
    colors_tw = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    mode_str = "Gel" if label == "gel" else "Liquid"
    eps_val  = EPS_GEL if label == "gel" else EPS_LIQ

    for ax_idx, (ax, logscale) in enumerate(zip(axes, [False, True])):
        for i, (tw, arr) in enumerate(msd_data):
            if len(arr) < 2:
                continue
            lags = arr[:, 0]
            msd  = arr[:, 1]
            col  = colors_tw[i % len(colors_tw)]
            ax.plot(lags, msd, lw=1.8, color=col,
                    label=f"tw = {tw:.0f} τ")
            # Diffusive reference D_free = kBT/γ → <Δr²> = 4 D_free t (2D)
            if i == 0 and ax_idx == 1 and len(lags) > 1:
                D_free = kBT / GAMMA
                t_ref  = np.array([lags[1], lags[-1]])
                ax.plot(t_ref, 4 * D_free * t_ref, 'k--', lw=1.2,
                        label="Free diffusion 4D₀t")

        if logscale:
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"MSD(Δt, tw) — {mode_str} [log-log]")
        else:
            ax.set_title(f"MSD(Δt, tw) — {mode_str} [linear]")
        ax.set_xlabel("Δt = t − tw  [τ]")
        ax.set_ylabel("MSD  [σ²]")
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Mean Squared Displacement — {mode_str} (ε = {eps_val} kBT)\n"
        f"Ageing: MSD depends on tw (gel) vs collapse (liquid)",
        fontsize=11
    )
    fig.tight_layout()
    savefig(fig, f"07_MSD_{label}.png")


# ── 8. MSD ageing comparison: gel vs liquid ────────────────
def plot_msd_comparison(msd_gel, msd_liq):
    """
    Side-by-side log-log MSD for all tw, gel vs liquid.
    The spread of curves = ageing strength.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             sharex=False, sharey=False)
    colors_tw = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for ax, msd_data, color_base, mode_str, eps_val in [
        (axes[0], msd_gel, C_GEL, "Gel",    EPS_GEL),
        (axes[1], msd_liq, C_LIQ, "Liquid", EPS_LIQ),
    ]:
        for i, (tw, arr) in enumerate(msd_data):
            if len(arr) < 2:
                continue
            lags = arr[:, 0]
            msd  = arr[:, 1]
            col  = colors_tw[i % len(colors_tw)]
            ax.plot(lags, msd, lw=2, color=col, label=f"tw = {tw:.0f} τ")

        # Free-diffusion reference
        D_free = kBT / GAMMA
        all_lags = np.concatenate([arr[:, 0] for _, arr in msd_data
                                   if len(arr) > 1])
        t_ref = np.array([all_lags[all_lags > 0].min(), all_lags.max()])
        ax.plot(t_ref, 4 * D_free * t_ref, 'k--', lw=1.2,
                label="Free diffusion 4D₀t")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Δt = t − tw  [τ]")
        ax.set_ylabel("MSD  [σ²]")
        ax.set_title(f"{mode_str} (ε = {eps_val} kBT)")
        ax.legend(fontsize=9)

    fig.suptitle(
        "MSD ageing comparison — Gel (left) vs Liquid (right)  [log-log]\n"
        "Gel: tw-dependent MSD → non-ergodic ageing.  "
        "Liquid: curves collapse → ergodic.",
        fontsize=11
    )
    fig.tight_layout()
    savefig(fig, "08_MSD_ageing_comparison.png")


# ── 9. MSD at fixed lag Δt = const, vs tw ─────────────────
def plot_msd_vs_tw(msd_gel, msd_liq):
    """
    MSD(Δt_fixed, tw) as a function of tw.
    Gel: decreasing MSD with tw → slowing dynamics (ageing).
    Liquid: flat → stationary (ergodic).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    def extract_at_lag(msd_data, target_frac=0.1):
        """Pick MSD at a lag ≈ target_frac × t_final for each tw."""
        t_target = target_frac * N_STEPS * DT
        tw_vals, msd_vals = [], []
        for tw, arr in msd_data:
            if len(arr) < 2:
                continue
            lags = arr[:, 0]
            msds = arr[:, 1]
            # find closest lag to t_target
            idx = np.argmin(np.abs(lags - t_target))
            tw_vals.append(tw)
            msd_vals.append(msds[idx])
        return np.array(tw_vals), np.array(msd_vals)

    for msd_data, color, marker, mode_str in [
        (msd_gel, C_GEL, 'o-', "Gel"),
        (msd_liq, C_LIQ, 's--', "Liquid"),
    ]:
        tw_arr, msd_arr = extract_at_lag(msd_data, target_frac=0.1)
        if len(tw_arr) > 0:
            ax.plot(tw_arr, msd_arr, marker, color=color, lw=2, ms=8,
                    label=mode_str)

    lag_shown = 0.1 * N_STEPS * DT
    ax.set_xlabel("Waiting time tw  [τ]")
    ax.set_ylabel(f"MSD(Δt ≈ {lag_shown:.0f} τ, tw)  [σ²]")
    ax.set_title(
        f"MSD at fixed lag Δt ≈ {lag_shown:.0f} τ  vs waiting time tw\n"
        "Gel: decreasing MSD → ageing dynamics.  Liquid: flat → ergodic."
    )
    ax.legend()
    fig.tight_layout()
    savefig(fig, "09_MSD_vs_tw.png")



def plot_dashboard(snaps_gel, snaps_liq, pos_gel, pos_liq, L,
                   pos_snaps_gel, pos_snaps_liq,
                   msd_gel=None, msd_liq=None):
    colors_tw = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    fig = plt.figure(figsize=(18, 16), facecolor="white")
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.52, wspace=0.35,
                            left=0.06, right=0.97, top=0.93, bottom=0.05)

    # Row 0: gel position snapshots
    for col, (t_s, p_s) in enumerate(pos_snaps_gel[:3]):
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(p_s[:,0], p_s[:,1], s=6, color=C_GEL, alpha=0.7, linewidths=0)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_title(f"Gel — t = {t_s:.0f} τ")
        ax.set_xlabel("x [σ]"); ax.set_ylabel("y [σ]")

    # Row 1: S(k) gel, S(k) liquid, comparison
    ax_sg = fig.add_subplot(gs[1, 0])
    ns = len(snaps_gel)
    for i, (t, k, S) in enumerate(snaps_gel):
        col = C_TIME[i % len(C_TIME)]
        lbl = f"t={t:.0f}τ" if i in [0, ns//2, ns-1] else "_"
        ax_sg.plot(k, S, color=col, lw=1.4, alpha=0.9, label=lbl)
        ax_sg.plot(k[np.argmax(S)], S[np.argmax(S)], 'o', color=col, ms=4)
    ax_sg.set_title(f"S(k) — Gel (ε={EPS_GEL}kBT)  [○ = k*]")
    ax_sg.set_xlabel("k [σ⁻¹]"); ax_sg.set_ylabel("S(k)")
    ax_sg.legend(fontsize=7)

    ax_sl = fig.add_subplot(gs[1, 1])
    ns2 = len(snaps_liq)
    for i, (t, k, S) in enumerate(snaps_liq):
        col = C_TIME[i % len(C_TIME)]
        lbl = f"t={t:.0f}τ" if i in [0, ns2//2, ns2-1] else "_"
        ax_sl.plot(k, S, color=col, lw=1.4, alpha=0.9, label=lbl)
        ax_sl.plot(k[np.argmax(S)], S[np.argmax(S)], 'o', color=col, ms=4)
    ax_sl.set_title(f"S(k) — Liquid (ε={EPS_LIQ}kBT)  [○ = k*]")
    ax_sl.set_xlabel("k [σ⁻¹]"); ax_sl.set_ylabel("S(k)")
    ax_sl.legend(fontsize=7)

    ax_sc = fig.add_subplot(gs[1, 2])
    tg, kg, Sg = snaps_gel[-1]; tl, kl, Sl = snaps_liq[-1]
    ax_sc.plot(kg, Sg, color=C_GEL, lw=2, label=f"Gel t={tg:.0f}τ")
    ax_sc.plot(kl, Sl, color=C_LIQ, lw=2, ls="--", label=f"Liquid t={tl:.0f}τ")
    ax_sc.set_title("Final S(k): Gel vs Liquid")
    ax_sc.set_xlabel("k [σ⁻¹]"); ax_sc.set_ylabel("S(k)")
    ax_sc.legend()

    # Row 2: k*(t), S(k*,t), final positions
    ax_ks = fig.add_subplot(gs[2, 0])
    t_g  = [s[0] for s in snaps_gel];  ks_g = [s[1][np.argmax(s[2])] for s in snaps_gel]
    t_l  = [s[0] for s in snaps_liq]; ks_l = [s[1][np.argmax(s[2])] for s in snaps_liq]
    ax_ks.plot(t_g, ks_g, 'o-', color=C_GEL, lw=1.8, ms=5, label="Gel")
    ax_ks.plot(t_l, ks_l, 's--', color=C_LIQ, lw=1.8, ms=5, label="Liquid")
    ax_ks.set_title("k*(t) — Arrest: k* plateaus"); ax_ks.set_xlabel("t [τ]"); ax_ks.set_ylabel("k* [σ⁻¹]")
    ax_ks.legend()

    ax_ss = fig.add_subplot(gs[2, 1])
    Ss_g = [np.max(s[2]) for s in snaps_gel]; Ss_l = [np.max(s[2]) for s in snaps_liq]
    ax_ss.plot(t_g, Ss_g, 'o-', color=C_GEL, lw=1.8, ms=5, label="Gel")
    ax_ss.plot(t_l, Ss_l, 's--', color=C_LIQ, lw=1.8, ms=5, label="Liquid")
    ax_ss.set_title("S(k*, t) — Coarsening: growing S(k*)"); ax_ss.set_xlabel("t [τ]"); ax_ss.set_ylabel("S(k*)")
    ax_ss.legend()

    ax_pf = fig.add_subplot(gs[2, 2])
    ax_pf.scatter(pos_gel[:,0], pos_gel[:,1], s=6, color=C_GEL, alpha=0.8,
                  linewidths=0, label="Gel")
    ax_pf.scatter(pos_liq[:,0], pos_liq[:,1], s=6, color=C_LIQ, alpha=0.5,
                  linewidths=0, label="Liquid")
    ax_pf.set_xlim(0, L); ax_pf.set_ylim(0, L); ax_pf.set_aspect("equal")
    ax_pf.set_title("Final positions: Gel (blue) vs Liquid (red)")
    ax_pf.set_xlabel("x [σ]"); ax_pf.set_ylabel("y [σ]")
    ax_pf.legend(fontsize=8)

    # Row 3: MSD ageing — gel (log-log), liquid (log-log), MSD vs tw
    if msd_gel is not None and msd_liq is not None:
        ax_mg = fig.add_subplot(gs[3, 0])
        for i, (tw, arr) in enumerate(msd_gel):
            if len(arr) < 2: continue
            ax_mg.plot(arr[:, 0], arr[:, 1], lw=1.8,
                       color=colors_tw[i % len(colors_tw)],
                       label=f"tw={tw:.0f}τ")
        ax_mg.set_xscale("log"); ax_mg.set_yscale("log")
        ax_mg.set_title("MSD — Gel [log-log]  (ageing → spread)")
        ax_mg.set_xlabel("Δt [τ]"); ax_mg.set_ylabel("MSD [σ²]")
        ax_mg.legend(fontsize=7)

        ax_ml = fig.add_subplot(gs[3, 1])
        for i, (tw, arr) in enumerate(msd_liq):
            if len(arr) < 2: continue
            ax_ml.plot(arr[:, 0], arr[:, 1], lw=1.8,
                       color=colors_tw[i % len(colors_tw)],
                       label=f"tw={tw:.0f}τ")
        ax_ml.set_xscale("log"); ax_ml.set_yscale("log")
        ax_ml.set_title("MSD — Liquid [log-log]  (ergodic → collapse)")
        ax_ml.set_xlabel("Δt [τ]"); ax_ml.set_ylabel("MSD [σ²]")
        ax_ml.legend(fontsize=7)

        ax_mtw = fig.add_subplot(gs[3, 2])
        t_target = 0.1 * N_STEPS * DT
        for msd_data, color, marker, mode_str in [
            (msd_gel, C_GEL, 'o-', "Gel"),
            (msd_liq, C_LIQ, 's--', "Liquid"),
        ]:
            tw_arr, msd_arr = [], []
            for tw, arr in msd_data:
                if len(arr) < 2: continue
                idx = np.argmin(np.abs(arr[:, 0] - t_target))
                tw_arr.append(tw); msd_arr.append(arr[idx, 1])
            if tw_arr:
                ax_mtw.plot(tw_arr, msd_arr, marker, color=color,
                            lw=1.8, ms=7, label=mode_str)
        ax_mtw.set_xlabel("tw [τ]")
        ax_mtw.set_ylabel(f"MSD(Δt≈{t_target:.0f}τ, tw) [σ²]")
        ax_mtw.set_title("MSD at fixed lag vs tw\nGel: decreasing → ageing")
        ax_mtw.legend(fontsize=8)

    fig.suptitle(
        f"Brownian Dynamics — Coarsening, Arrest & Ageing in Colloidal Gels  [v8]\n"
        f"N={N}   φ={PHI}   dt={DT}   ε_gel={EPS_GEL}kBT   ε_liq={EPS_LIQ}kBT"
        f"   t_final={N_STEPS*DT:.0f}τ   tw={[f'{f*N_STEPS*DT:.0f}τ' for f in TW_FRACS]}",
        fontsize=10
    )
    savefig(fig, "00_dashboard.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print(f"  N={N}, steps={N_STEPS}, dt={DT}, t_final={N_STEPS*DT:.0f}τ")
    print(f"  ε_gel={EPS_GEL}kBT  ε_liq={EPS_LIQ}kBT  force_cap={FORCE_CAP}")
    print(f"  Ageing: tw = {[f*N_STEPS*DT for f in TW_FRACS]} τ")
    print(f"  Output: ./{PLOT_DIR}/*.png")
    print("="*60)

    snaps_gel, pos_gel, L, ps_gel, msd_gel = run_simulation(mode="gel")
    snaps_liq, pos_liq, _,  ps_liq, msd_liq = run_simulation(mode="liquid")

    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"\n  Saving plots to ./{PLOT_DIR}/")

    plot_positions_snapshots(ps_gel,  L, label="gel")
    plot_positions_snapshots(ps_liq,  L, label="liquid")
    plot_sk_evolution(snaps_gel, label="gel")
    plot_sk_evolution(snaps_liq, label="liquid")
    plot_sk_comparison(snaps_gel, snaps_liq)
    plot_kstar(snaps_gel, snaps_liq)
    plot_sstar(snaps_gel, snaps_liq)
    plot_final_positions(pos_gel, pos_liq, L)
    # ── New: ageing / MSD plots ────────────────────────────
    plot_msd(msd_gel, label="gel")
    plot_msd(msd_liq, label="liquid")
    plot_msd_comparison(msd_gel, msd_liq)
    plot_msd_vs_tw(msd_gel, msd_liq)
    # ── Dashboard (now includes MSD row) ──────────────────
    plot_dashboard(snaps_gel, snaps_liq, pos_gel, pos_liq, L,
                   ps_gel, ps_liq, msd_gel, msd_liq)

    print(f"\n  Done — {len(os.listdir(PLOT_DIR))} files in ./{PLOT_DIR}/")
