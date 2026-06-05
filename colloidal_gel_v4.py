"""
=============================================================
 Dynamique Brownienne — Coarsening & Arrêt dans les Gels Colloïdaux
 v4 : forces compilées avec numba (@njit) → stable et rapide
=============================================================
Équation de Langevin overdamped, schéma Euler-Maruyama :
    r_i(t+dt) = r_i(t) + (dt/γ) F_i + sqrt(2 kBT dt/γ) η_i

Potentiels :
  - SANS gel : WCA (LJ coupé au minimum r_c = 2^(1/6)σ, purement répulsif)
  - AVEC gel  : LJ attractif coupé à r_c = 2.5σ, ε = 5 kBT

Conditions aux limites : périodiques (PBC), boîte carrée de côté L
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from numba import njit

# ─────────────────────────────────────────────
#  PARAMÈTRES
# ─────────────────────────────────────────────
N          = 300
PHI        = 0.25
SIGMA      = 1.0
kBT        = 1.0
GAMMA      = 1.0

EPS_GEL    = 5.0
EPS_WCA    = 1.0

DT         = 5e-4         # dt petit → stable même sans clamp
N_EQUIL    = 5_000        # équilibration douce
N_STEPS    = 30_000       # t_final = 15 τ
SNAP_EVERY = 3_000

NK_SHELLS  = 30
NK_MAX     = 20
SEED       = 42

R_CUT_GEL  = 2.5
R_CUT_WCA  = 2**(1/6)

# ─────────────────────────────────────────────
#  FORCES — compilées numba, boucle sur paires
#  O(N²) mais exécuté en C → très rapide
# ─────────────────────────────────────────────

@njit(cache=True)
def compute_forces_numba(pos, L, eps, r_cut):
    """
    Boucle sur toutes les paires (i<j), PBC, LJ shifté au cutoff.
    Retourne forces (N,2) et énergie potentielle U.
    """
    N      = pos.shape[0]
    forces = np.zeros((N, 2))
    U_tot  = 0.0

    sig2   = SIGMA**2
    r_cut2 = r_cut**2

    # Shift au cutoff pour continuité de U
    ir2c   = sig2 / r_cut2
    ir6c   = ir2c**3
    U_cut  = 4.0 * eps * (ir6c*ir6c - ir6c)

    for i in range(N - 1):
        for j in range(i + 1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]

            # PBC image minimale
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)

            r2 = dx*dx + dy*dy
            if r2 >= r_cut2 or r2 < 1e-12:
                continue

            ir2  = sig2 / r2
            ir6  = ir2 * ir2 * ir2
            ir12 = ir6 * ir6

            U = 4.0 * eps * (ir12 - ir6) - U_cut
            U_tot += U

            # Force : f/r² * dr  (Newton 3)
            fr = 24.0 * eps / r2 * (2.0*ir12 - ir6)
            fx = fr * dx
            fy = fr * dy

            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[j, 0] -= fx
            forces[j, 1] -= fy

    return forces, U_tot


# ─────────────────────────────────────────────
#  INIT POSITIONS
# ─────────────────────────────────────────────

def init_positions(N, phi, sigma, seed=42):
    L = np.sqrt(N * np.pi * (sigma / 2)**2 / phi)
    rng = np.random.default_rng(seed)
    n_side = int(np.ceil(np.sqrt(N)))
    spacing = L / n_side
    xs = (np.arange(n_side) + 0.5) * spacing
    grid = np.array([[x, y] for y in xs for x in xs])[:N]
    pos = grid + rng.uniform(-0.05*spacing, 0.05*spacing, size=(N, 2))
    return pos % L, L


# ─────────────────────────────────────────────
#  ÉQUILIBRATION DOUCE (WCA, ε croissant)
# ─────────────────────────────────────────────

def equilibrate(pos, L, n_steps, seed):
    rng = np.random.default_rng(seed + 99)
    dt_eq = 1e-4   # très petit pendant l'équilibration
    print(f"  Équilibration ({n_steps} pas, dt={dt_eq}) ...", end="", flush=True)
    for step in range(n_steps):
        # ε monte progressivement 0.1 → 1.0 kBT sur la première moitié
        frac    = min(1.0, (step + 1) / (n_steps * 0.5))
        eps_now = 0.1 + 0.9 * frac
        forces, _ = compute_forces_numba(pos, L, eps_now, R_CUT_WCA)
        noise  = rng.standard_normal(pos.shape)
        amp    = np.sqrt(2.0 * kBT * dt_eq / GAMMA)
        pos    = (pos + (dt_eq / GAMMA) * forces + amp * noise) % L
    print(" OK")
    return pos


# ─────────────────────────────────────────────
#  PAS EULER-MARUYAMA
# ─────────────────────────────────────────────

def em_step(pos, L, eps, r_cut, rng):
    forces, U = compute_forces_numba(pos, L, eps, r_cut)
    noise     = rng.standard_normal(pos.shape)
    amp       = np.sqrt(2.0 * kBT * DT / GAMMA)
    pos_new   = (pos + (DT / GAMMA) * forces + amp * noise) % L
    return pos_new, U


# ─────────────────────────────────────────────
#  FACTEUR DE STRUCTURE S(k)
# ─────────────────────────────────────────────

def structure_factor(pos, L, nk_shells=NK_SHELLS, nk_max=NK_MAX):
    k0 = 2.0 * np.pi / L
    n  = len(pos)

    nx_arr, ny_arr = np.meshgrid(np.arange(-nk_max, nk_max+1),
                                  np.arange(-nk_max, nk_max+1))
    nx_arr = nx_arr.ravel()
    ny_arr = ny_arr.ravel()
    n2     = nx_arr**2 + ny_arr**2
    sel    = (n2 >= 1) & (n2 <= nk_max**2)
    nx_arr, ny_arr, n2 = nx_arr[sel], ny_arr[sel], n2[sel]

    kx     = nx_arr * k0
    ky     = ny_arr * k0
    k_mags = np.sqrt(n2) * k0

    phase  = kx[:, None] * pos[:, 0] + ky[:, None] * pos[:, 1]
    Sk_all = np.abs(np.sum(np.exp(1j * phase), axis=1))**2 / n

    bins      = np.linspace(k_mags.min()*0.99, k_mags.max()*1.01, nk_shells+1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    idx       = np.searchsorted(bins, k_mags, side='right') - 1
    S_binned  = np.zeros(nk_shells)
    counts    = np.zeros(nk_shells)
    np.add.at(S_binned, idx, Sk_all)
    np.add.at(counts,   idx, 1)
    valid = counts > 0
    S_binned[valid] /= counts[valid]

    return k_centers[valid], S_binned[valid]


# ─────────────────────────────────────────────
#  SIMULATION PRINCIPALE
# ─────────────────────────────────────────────

def run_simulation(mode="gel", seed=SEED):
    rng = np.random.default_rng(seed)
    pos, L = init_positions(N, PHI, SIGMA, seed=seed)

    label = "GÉLATION (LJ attractif, ε=5kBT)" if mode=="gel" \
            else "SANS GEL (WCA, purement répulsif)"
    print(f"\n{'='*55}\n  Mode : {label}")

    # Compilation numba au premier appel (quelques secondes)
    print("  Compilation numba (premier appel) ...", end="", flush=True)
    _ = compute_forces_numba(pos[:10], L, 1.0, R_CUT_WCA)
    print(" OK")

    pos = equilibrate(pos, L, N_EQUIL, seed)

    eps   = EPS_GEL   if mode == "gel" else EPS_WCA
    r_cut = R_CUT_GEL if mode == "gel" else R_CUT_WCA
    t_final = N_STEPS * DT

    print(f"  N={N}, φ={PHI}, dt={DT}, steps={N_STEPS}, t_final={t_final:.1f} τ")
    print(f"  ε={eps:.1f} kBT, r_cut={r_cut:.3f} σ")

    snapshots, energies = [], []

    for step in range(N_STEPS + 1):
        if step % SNAP_EVERY == 0:
            t = step * DT
            k_vals, S_vals = structure_factor(pos, L)
            snapshots.append((t, k_vals.copy(), S_vals.copy()))
            _, U = compute_forces_numba(pos, L, eps, r_cut)
            energies.append((t, U))
            print(f"  step {step:6d}/{N_STEPS}  t={t:5.2f} τ  U={U:10.2f} kBT")
        if step < N_STEPS:
            pos, _ = em_step(pos, L, eps, r_cut, rng)

    return snapshots, energies, pos, L


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def plot_results(snaps_gel, snaps_wca, pos_gel, pos_wca, L, en_gel, en_wca):
    cg = LinearSegmentedColormap.from_list("g", ["#2d0040", "#a020f0", "#ff6b35"])
    cw = LinearSegmentedColormap.from_list("w", ["#001a33", "#0066cc", "#66ddff"])

    fig = plt.figure(figsize=(17, 10), facecolor="#090909")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.42, wspace=0.33,
                            left=0.06, right=0.97, top=0.91, bottom=0.08)
    ax_pg = fig.add_subplot(gs[0, 0])
    ax_sg = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[0, 2])
    ax_pw = fig.add_subplot(gs[1, 0])
    ax_sw = fig.add_subplot(gs[1, 1])
    ax_en = fig.add_subplot(gs[1, 2])

    tkw = dict(color="white", fontfamily="monospace", fontsize=9, pad=5)

    # ── Positions ──────────────────────────────────
    for ax, pos, title, cm in [
        (ax_pg, pos_gel, "Positions finales — GEL",     cg),
        (ax_pw, pos_wca, "Positions finales — SANS GEL", cw),
    ]:
        ax.set_facecolor("#0d0d0d")
        ax.scatter(pos[:,0], pos[:,1], c=pos[:,0]+pos[:,1],
                   cmap=cm, s=10, alpha=0.9, linewidths=0)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_title(title, **tkw)
        ax.tick_params(colors="#555", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#222")

    # ── S(k) gel ───────────────────────────────────
    ns = len(snaps_gel)
    ax_sg.set_facecolor("#0d0d0d")
    for i, (t, k, S) in enumerate(snaps_gel):
        col = cg(i / max(ns-1, 1))
        lbl = f"t={t:.1f}τ" if i in [0, ns//2, ns-1] else ""
        ax_sg.plot(k, S, color=col, lw=1.6, alpha=0.9, label=lbl)
    ax_sg.set_title("S(k) — GEL", **tkw)
    ax_sg.set_xlabel("k [σ⁻¹]", color="#888", fontsize=8)
    ax_sg.set_ylabel("S(k)", color="#888", fontsize=8)
    ax_sg.tick_params(colors="#555", labelsize=7)
    ax_sg.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
    for sp in ax_sg.spines.values(): sp.set_edgecolor("#222")

    # ── S(k) WCA ───────────────────────────────────
    ns2 = len(snaps_wca)
    ax_sw.set_facecolor("#0d0d0d")
    for i, (t, k, S) in enumerate(snaps_wca):
        col = cw(i / max(ns2-1, 1))
        lbl = f"t={t:.1f}τ" if i in [0, ns2//2, ns2-1] else ""
        ax_sw.plot(k, S, color=col, lw=1.6, alpha=0.9, label=lbl)
    ax_sw.set_title("S(k) — SANS GEL / WCA", **tkw)
    ax_sw.set_xlabel("k [σ⁻¹]", color="#888", fontsize=8)
    ax_sw.set_ylabel("S(k)", color="#888", fontsize=8)
    ax_sw.tick_params(colors="#555", labelsize=7)
    ax_sw.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
    for sp in ax_sw.spines.values(): sp.set_edgecolor("#222")

    # ── Comparaison finale ─────────────────────────
    ax_sc.set_facecolor("#0d0d0d")
    tg, kg, Sg = snaps_gel[-1]
    tw, kw2, Sw = snaps_wca[-1]
    ax_sc.plot(kg,  Sg, color="#ff6b35", lw=2.2, label=f"Gel  t={tg:.1f}τ")
    ax_sc.plot(kw2, Sw, color="#66ddff", lw=2.2, ls="--", label=f"WCA  t={tw:.1f}τ")
    ax_sc.set_title("S(k) final : GEL vs SANS GEL", **tkw)
    ax_sc.set_xlabel("k [σ⁻¹]", color="#888", fontsize=8)
    ax_sc.set_ylabel("S(k)", color="#888", fontsize=8)
    ax_sc.tick_params(colors="#555", labelsize=7)
    ax_sc.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", framealpha=0.7)
    for sp in ax_sc.spines.values(): sp.set_edgecolor("#222")

    # ── Énergie ────────────────────────────────────
    ax_en.set_facecolor("#0d0d0d")
    tg_e, Ug = zip(*en_gel)
    tw_e, Uw = zip(*en_wca)
    ax_en.plot(tg_e, Ug, color="#ff6b35", lw=1.8, label="Gel (LJ attractif)")
    ax_en.plot(tw_e, Uw, color="#66ddff", lw=1.8, ls="--", label="WCA (répulsif)")
    ax_en.set_title("Énergie potentielle U(t)", **tkw)
    ax_en.set_xlabel("t [τ]", color="#888", fontsize=8)
    ax_en.set_ylabel("U [kBT]", color="#888", fontsize=8)
    ax_en.tick_params(colors="#555", labelsize=7)
    ax_en.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", framealpha=0.7)
    for sp in ax_en.spines.values(): sp.set_edgecolor("#222")

    fig.suptitle(
        f"Dynamique Brownienne — Coarsening & Arrêt dans les Gels Colloïdaux\n"
        f"N={N}  φ={PHI}  dt={DT}  ε_gel={EPS_GEL:.0f}kBT"
        f"  t_final={N_STEPS*DT:.1f}τ  (Euler-Maruyama + numba)",
        color="white", fontsize=10, fontfamily="monospace", y=0.97
    )
    plt.savefig("colloidal_gel_v4.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\n  Figure sauvegardée : colloidal_gel_v4.png")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print(f"  N={N}, {N_STEPS} pas, dt={DT}, t_final={N_STEPS*DT:.1f} τ")
    print(f"  Forces numba @njit — ~10-15 min total")
    print("="*55)

    snaps_gel, en_gel, pos_gel, L = run_simulation(mode="gel")
    snaps_wca, en_wca, pos_wca, _ = run_simulation(mode="no_gel")
    plot_results(snaps_gel, snaps_wca, pos_gel, pos_wca, L, en_gel, en_wca)
