"""
=============================================================
 Dynamique Brownienne — Coarsening & Arrêt dans les Gels Colloïdaux
 v5 : dt=2e-4, EPS_GEL=8kBT, t_final=10τ, affichage k* et S(k*) terminal
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

EPS_GEL    = 8.0          # puits attractif plus profond → gel plus franc
EPS_WCA    = 1.0

DT         = 2e-4         # plus petit → stable pour EPS=8kBT
N_EQUIL    = 5_000        # équilibration douce
N_STEPS    = 50_000       # t_final = 10 τ
SNAP_EVERY = 5_000        # 10 snapshots

NK_SHELLS  = 30
NK_MAX     = 20
SEED       = 42

R_CUT_GEL  = 2.5
R_CUT_WCA  = 2**(1/6)

# ─────────────────────────────────────────────
#  FORCES NUMBA
# ─────────────────────────────────────────────

@njit(cache=True)
def compute_forces_numba(pos, L, eps, r_cut):
    N      = pos.shape[0]
    forces = np.zeros((N, 2))
    U_tot  = 0.0
    sig2   = SIGMA**2
    r_cut2 = r_cut**2
    ir2c   = sig2 / r_cut2
    ir6c   = ir2c**3
    U_cut  = 4.0 * eps * (ir6c*ir6c - ir6c)

    for i in range(N - 1):
        for j in range(i + 1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            r2 = dx*dx + dy*dy
            if r2 >= r_cut2 or r2 < 1e-12:
                continue
            ir2  = sig2 / r2
            ir6  = ir2 * ir2 * ir2
            ir12 = ir6 * ir6
            U_tot += 4.0 * eps * (ir12 - ir6) - U_cut
            fr = 24.0 * eps / r2 * (2.0*ir12 - ir6)
            fx = fr * dx
            fy = fr * dy
            forces[i, 0] += fx;  forces[i, 1] += fy
            forces[j, 0] -= fx;  forces[j, 1] -= fy

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
#  ÉQUILIBRATION DOUCE
# ─────────────────────────────────────────────

def equilibrate(pos, L, n_steps, seed):
    rng = np.random.default_rng(seed + 99)
    dt_eq = 1e-4
    print(f"  Équilibration ({n_steps} pas, dt={dt_eq}) ...", end="", flush=True)
    for step in range(n_steps):
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
    noise   = rng.standard_normal(pos.shape)
    amp     = np.sqrt(2.0 * kBT * DT / GAMMA)
    pos_new = (pos + (DT / GAMMA) * forces + amp * noise) % L
    return pos_new, U


# ─────────────────────────────────────────────
#  FACTEUR DE STRUCTURE S(k)
# ─────────────────────────────────────────────

def structure_factor(pos, L, nk_shells=NK_SHELLS, nk_max=NK_MAX):
    k0  = 2.0 * np.pi / L
    n   = len(pos)
    nx_arr, ny_arr = np.meshgrid(np.arange(-nk_max, nk_max+1),
                                  np.arange(-nk_max, nk_max+1))
    nx_arr = nx_arr.ravel();  ny_arr = ny_arr.ravel()
    n2     = nx_arr**2 + ny_arr**2
    sel    = (n2 >= 1) & (n2 <= nk_max**2)
    nx_arr, ny_arr, n2 = nx_arr[sel], ny_arr[sel], n2[sel]

    kx     = nx_arr * k0;  ky = ny_arr * k0
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
#  AFFICHAGE TERMINAL DES VALEURS S(k)
# ─────────────────────────────────────────────

def print_sk_table(t, k_vals, S_vals, U, mode):
    """Affiche un tableau k / S(k) dans le terminal à chaque snapshot."""
    i_peak = np.argmax(S_vals)
    k_star = k_vals[i_peak]
    S_star = S_vals[i_peak]

    print(f"\n  ── Snapshot t={t:.2f} τ  [{mode}]  U={U:.2f} kBT ──")
    print(f"  {'k [σ⁻¹]':>10}  {'S(k)':>10}")
    print(f"  {'-'*23}")
    for k, S in zip(k_vals, S_vals):
        marker = "  ← pic" if abs(k - k_star) < 1e-6 else ""
        print(f"  {k:10.4f}  {S:10.4f}{marker}")
    print(f"  → k* = {k_star:.4f} σ⁻¹   S(k*) = {S_star:.4f}")


# ─────────────────────────────────────────────
#  SIMULATION PRINCIPALE
# ─────────────────────────────────────────────

def run_simulation(mode="gel", seed=SEED):
    rng = np.random.default_rng(seed)
    pos, L = init_positions(N, PHI, SIGMA, seed=seed)

    label = "GÉLATION (LJ attractif, ε=8kBT)" if mode=="gel" \
            else "SANS GEL (WCA, purement répulsif)"
    print(f"\n{'='*55}\n  Mode : {label}")

    print("  Compilation numba ...", end="", flush=True)
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
            _, U = compute_forces_numba(pos, L, eps, r_cut)

            # ── Affichage terminal ──
            print_sk_table(t, k_vals, S_vals, U, mode)

            snapshots.append((t, k_vals.copy(), S_vals.copy()))
            energies.append((t, U))

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

    # ── Positions ──
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

    # ── S(k) gel ──
    ns = len(snaps_gel)
    ax_sg.set_facecolor("#0d0d0d")
    for i, (t, k, S) in enumerate(snaps_gel):
        col = cg(i / max(ns-1, 1))
        lbl = f"t={t:.1f}τ" if i in [0, ns//2, ns-1] else ""
        ax_sg.plot(k, S, color=col, lw=1.6, alpha=0.9, label=lbl)
        # Marquer le pic
        i_pk = np.argmax(S)
        ax_sg.plot(k[i_pk], S[i_pk], 'o', color=col, ms=4)
    ax_sg.set_title("S(k) — GEL  (o = pic k*)", **tkw)
    ax_sg.set_xlabel("k [σ⁻¹]", color="#888", fontsize=8)
    ax_sg.set_ylabel("S(k)", color="#888", fontsize=8)
    ax_sg.tick_params(colors="#555", labelsize=7)
    ax_sg.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
    for sp in ax_sg.spines.values(): sp.set_edgecolor("#222")

    # ── S(k) WCA ──
    ns2 = len(snaps_wca)
    ax_sw.set_facecolor("#0d0d0d")
    for i, (t, k, S) in enumerate(snaps_wca):
        col = cw(i / max(ns2-1, 1))
        lbl = f"t={t:.1f}τ" if i in [0, ns2//2, ns2-1] else ""
        ax_sw.plot(k, S, color=col, lw=1.6, alpha=0.9, label=lbl)
        i_pk = np.argmax(S)
        ax_sw.plot(k[i_pk], S[i_pk], 'o', color=col, ms=4)
    ax_sw.set_title("S(k) — SANS GEL / WCA  (o = pic k*)", **tkw)
    ax_sw.set_xlabel("k [σ⁻¹]", color="#888", fontsize=8)
    ax_sw.set_ylabel("S(k)", color="#888", fontsize=8)
    ax_sw.tick_params(colors="#555", labelsize=7)
    ax_sw.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
    for sp in ax_sw.spines.values(): sp.set_edgecolor("#222")

    # ── Comparaison finale ──
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

    # ── k* et S(k*) en fonction du temps ──
    ax_en.set_facecolor("#0d0d0d")

    # Extraire k* et S(k*) à chaque snapshot
    t_gel  = [s[0] for s in snaps_gel]
    kstar_gel = [s[1][np.argmax(s[2])] for s in snaps_gel]
    Sstar_gel = [np.max(s[2]) for s in snaps_gel]

    t_wca  = [s[0] for s in snaps_wca]
    kstar_wca = [s[1][np.argmax(s[2])] for s in snaps_wca]
    Sstar_wca = [np.max(s[2]) for s in snaps_wca]

    ax_en2 = ax_en.twinx()
    l1, = ax_en.plot(t_gel, kstar_gel, 'o-', color="#ff6b35", lw=1.8, ms=5, label="k* gel")
    l2, = ax_en.plot(t_wca, kstar_wca, 's--', color="#66ddff", lw=1.8, ms=5, label="k* WCA")
    l3, = ax_en2.plot(t_gel, Sstar_gel, 'o-', color="#ffaa00", lw=1.5, ms=4, alpha=0.7, label="S(k*) gel")
    l4, = ax_en2.plot(t_wca, Sstar_wca, 's--', color="#aaddff", lw=1.5, ms=4, alpha=0.7, label="S(k*) WCA")

    ax_en.set_title("k*(t) et S(k*,t) — arrêt = k* constant", **tkw)
    ax_en.set_xlabel("t [τ]", color="#888", fontsize=8)
    ax_en.set_ylabel("k* [σ⁻¹]", color="#ff6b35", fontsize=8)
    ax_en2.set_ylabel("S(k*)", color="#ffaa00", fontsize=8)
    ax_en.tick_params(colors="#555", labelsize=7)
    ax_en2.tick_params(colors="#555", labelsize=7)
    lines = [l1, l2, l3, l4]
    ax_en.legend(lines, [l.get_label() for l in lines],
                 fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
    for sp in ax_en.spines.values(): sp.set_edgecolor("#222")

    fig.suptitle(
        f"Dynamique Brownienne — Coarsening & Arrêt dans les Gels Colloïdaux\n"
        f"N={N}  φ={PHI}  dt={DT}  ε_gel={EPS_GEL:.0f}kBT"
        f"  t_final={N_STEPS*DT:.1f}τ  (Euler-Maruyama + numba)",
        color="white", fontsize=10, fontfamily="monospace", y=0.97
    )
    plt.savefig("colloidal_gel_v5.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\n  Figure sauvegardée : colloidal_gel_v5.png")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print(f"  N={N}, {N_STEPS} pas, dt={DT}, t_final={N_STEPS*DT:.1f} τ")
    print(f"  ε_gel={EPS_GEL} kBT  —  numba @njit")
    print(f"  ~15 min total")
    print("="*55)

    snaps_gel, en_gel, pos_gel, L = run_simulation(mode="gel")
    snaps_wca, en_wca, pos_wca, _ = run_simulation(mode="no_gel")
    plot_results(snaps_gel, snaps_wca, pos_gel, pos_wca, L, en_gel, en_wca)
