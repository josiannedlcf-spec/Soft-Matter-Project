"""
=============================================================
 Dynamique Brownienne — Coarsening & Arrêt dans les Gels Colloïdaux
 v6 : corrections physiques et numériques
 ─────────────────────────────────────────────────────────
 BUGS CORRIGÉS :
   1. DT trop grand pour ε=8kBT → réduit à 5e-5, avec cap sur ||force||
   2. t_final trop court (10τ) pour voir la gélation → 200τ
   3. Potentiel gel = WCA (cœur dur) + puits Morse/LJ tronqué :
      assure répulsion correcte + attraction contrôlée
   4. Équilibration sur WCA uniquement (correct), puis switch gel
   5. Structure factor : NK_MAX augmenté, normalisation corrigée
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

EPS_GEL    = 5.0          # puits attractif — 5kBT suffit pour gélation 2D
EPS_WCA    = 1.0

# ── FIX 1 : dt beaucoup plus petit ──
DT         = 5e-5         # stable pour ε=5kBT avec cap sur les forces

N_EQUIL    = 8_000        # équilibration WCA
# ── FIX 2 : t_final = 200τ ──
N_STEPS    = 4_000_000    # 4e6 × 5e-5 = 200 τ
SNAP_EVERY = 400_000      # 10 snapshots

NK_SHELLS  = 40
NK_MAX     = 25
SEED       = 42

R_CUT_GEL  = 2.5          # LJ tronqué à 2.5σ
R_CUT_WCA  = 2**(1/6)     # WCA = LJ tronqué au minimum

# ── FIX 3 : cap sur les forces pour éviter les explosions ──
FORCE_CAP  = 500.0        # en unités kBT/σ

# ─────────────────────────────────────────────
#  FORCES NUMBA  (avec cap)
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
            # ── FIX : distance minimale plus grande → évite divergence ──
            if r2 >= r_cut2 or r2 < 0.25:  # r_min = 0.5σ
                continue
            ir2  = sig2 / r2
            ir6  = ir2 * ir2 * ir2
            ir12 = ir6 * ir6
            U_tot += 4.0 * eps * (ir12 - ir6) - U_cut
            fr = 24.0 * eps / r2 * (2.0*ir12 - ir6)

            # ── FIX : cap sur la magnitude de la force ──
            r_inv = 1.0 / (r2**0.5)
            f_mag = abs(fr) * r2**0.5   # |F| = fr * r
            if f_mag > FORCE_CAP:
                fr = FORCE_CAP * r_inv * (1.0 if fr > 0 else -1.0) / r2**0.5

            fx = fr * dx
            fy = fr * dy
            forces[i, 0] += fx;  forces[i, 1] += fy
            forces[j, 0] -= fx;  forces[j, 1] -= fy

    return forces, U_tot


# ─────────────────────────────────────────────
#  INIT POSITIONS  (grille + bruit faible)
# ─────────────────────────────────────────────

def init_positions(N, phi, sigma, seed=42):
    L = np.sqrt(N * np.pi * (sigma / 2)**2 / phi)
    rng = np.random.default_rng(seed)
    n_side = int(np.ceil(np.sqrt(N)))
    spacing = L / n_side
    xs = (np.arange(n_side) + 0.5) * spacing
    grid = np.array([[x, y] for y in xs for x in xs])[:N]
    # ── FIX : bruit encore plus faible pour éviter chevauchements ──
    pos = grid + rng.uniform(-0.02*spacing, 0.02*spacing, size=(N, 2))
    return pos % L, L


# ─────────────────────────────────────────────
#  ÉQUILIBRATION  (WCA uniquement, rampe douce)
# ─────────────────────────────────────────────

def equilibrate(pos, L, n_steps, seed):
    rng = np.random.default_rng(seed + 99)
    dt_eq = 2e-5   # encore plus petit pendant l'équilibration
    print(f"  Équilibration ({n_steps} pas, dt={dt_eq}) ...", end="", flush=True)
    for step in range(n_steps):
        frac    = min(1.0, (step + 1) / (n_steps * 0.3))
        eps_now = 0.05 + 0.95 * frac   # rampe de 0.05→1.0 kBT
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
    # ── FIX : exclure n2=0 (k=0) et les harmoniques trop hautes ──
    sel    = (n2 >= 1) & (n2 <= nk_max**2)
    nx_arr, ny_arr, n2 = nx_arr[sel], ny_arr[sel], n2[sel]

    kx     = nx_arr * k0;  ky = ny_arr * k0
    k_mags = np.sqrt(n2) * k0

    # Calcul vectorisé : phase[i_k, i_particle]
    phase  = kx[:, None] * pos[:, 0] + ky[:, None] * pos[:, 1]
    Sk_all = np.abs(np.sum(np.exp(1j * phase), axis=1))**2 / n

    bins      = np.linspace(k_mags.min()*0.99, k_mags.max()*1.01, nk_shells+1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    idx       = np.searchsorted(bins, k_mags, side='right') - 1
    idx       = np.clip(idx, 0, nk_shells - 1)
    S_binned  = np.zeros(nk_shells)
    counts    = np.zeros(nk_shells)
    np.add.at(S_binned, idx, Sk_all)
    np.add.at(counts,   idx, 1)
    valid = counts > 0
    S_binned[valid] /= counts[valid]

    return k_centers[valid], S_binned[valid]


# ─────────────────────────────────────────────
#  AFFICHAGE TERMINAL
# ─────────────────────────────────────────────

def print_sk_table(t, k_vals, S_vals, U, mode):
    i_peak = np.argmax(S_vals)
    k_star = k_vals[i_peak]
    S_star = S_vals[i_peak]
    print(f"\n  ── t={t:.1f}τ [{mode}]  U={U:.1f}kBT  k*={k_star:.4f}σ⁻¹  S(k*)={S_star:.3f} ──")


# ─────────────────────────────────────────────
#  SIMULATION PRINCIPALE
# ─────────────────────────────────────────────

def run_simulation(mode="gel", seed=SEED):
    rng = np.random.default_rng(seed)
    pos, L = init_positions(N, PHI, SIGMA, seed=seed)

    label = f"GÉLATION (LJ, ε={EPS_GEL}kBT)" if mode=="gel" \
            else "SANS GEL (WCA, purement répulsif)"
    print(f"\n{'='*60}\n  Mode : {label}")
    print(f"  N={N}, φ={PHI}, dt={DT}, steps={N_STEPS}, t_final={N_STEPS*DT:.0f}τ")
    print(f"  L={L:.2f}σ")

    print("  Compilation numba ...", end="", flush=True)
    _ = compute_forces_numba(pos[:10], L, 1.0, R_CUT_WCA)
    print(" OK")

    pos = equilibrate(pos, L, N_EQUIL, seed)

    eps   = EPS_GEL   if mode == "gel" else EPS_WCA
    r_cut = R_CUT_GEL if mode == "gel" else R_CUT_WCA

    snapshots, energies, pos_snaps = [], [], []

    for step in range(N_STEPS + 1):
        if step % SNAP_EVERY == 0:
            t = step * DT
            k_vals, S_vals = structure_factor(pos, L)
            _, U = compute_forces_numba(pos, L, eps, r_cut)
            print_sk_table(t, k_vals, S_vals, U, mode)
            snapshots.append((t, k_vals.copy(), S_vals.copy()))
            energies.append((t, U))
            # Sauvegarder quelques snapshots de position
            if len(pos_snaps) < 4:
                pos_snaps.append((t, pos.copy()))

        if step < N_STEPS:
            pos, _ = em_step(pos, L, eps, r_cut, rng)

    return snapshots, energies, pos, L, pos_snaps


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def plot_results(snaps_gel, snaps_wca, pos_gel, pos_wca, L,
                 en_gel, en_wca, pos_snaps_gel, pos_snaps_wca):

    cg = LinearSegmentedColormap.from_list("g", ["#2d0040", "#a020f0", "#ff6b35"])
    cw = LinearSegmentedColormap.from_list("w", ["#001a33", "#0066cc", "#66ddff"])

    fig = plt.figure(figsize=(18, 12), facecolor="#090909")
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.45, wspace=0.33,
                            left=0.06, right=0.97, top=0.93, bottom=0.06)

    tkw = dict(color="white", fontfamily="monospace", fontsize=9, pad=5)

    # ── Ligne 0 : snapshots de positions gel ──
    titles_g = [f"Gel t={t:.0f}τ" for t, _ in pos_snaps_gel[:3]]
    for col, (t_s, p_s) in enumerate(pos_snaps_gel[:3]):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#0d0d0d")
        ax.scatter(p_s[:,0], p_s[:,1], c=p_s[:,0]+p_s[:,1],
                   cmap=cg, s=8, alpha=0.9, linewidths=0)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_title(f"GEL — t={t_s:.0f}τ", **tkw)
        ax.tick_params(colors="#555", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#222")

    # ── Ligne 1 : S(k) gel, S(k) WCA, comparaison finale ──
    ax_sg = fig.add_subplot(gs[1, 0])
    ax_sw = fig.add_subplot(gs[1, 1])
    ax_sc = fig.add_subplot(gs[1, 2])

    ns = len(snaps_gel)
    ax_sg.set_facecolor("#0d0d0d")
    for i, (t, k, S) in enumerate(snaps_gel):
        col = cg(i / max(ns-1, 1))
        lbl = f"t={t:.0f}τ" if i in [0, ns//2, ns-1] else ""
        ax_sg.plot(k, S, color=col, lw=1.6, alpha=0.9, label=lbl)
        i_pk = np.argmax(S)
        ax_sg.plot(k[i_pk], S[i_pk], 'o', color=col, ms=4)
    ax_sg.set_title("S(k) — GEL  (○=pic k*)", **tkw)
    ax_sg.set_xlabel("k [σ⁻¹]", color="#888", fontsize=8)
    ax_sg.set_ylabel("S(k)", color="#888", fontsize=8)
    ax_sg.tick_params(colors="#555", labelsize=7)
    ax_sg.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
    for sp in ax_sg.spines.values(): sp.set_edgecolor("#222")

    ns2 = len(snaps_wca)
    ax_sw.set_facecolor("#0d0d0d")
    for i, (t, k, S) in enumerate(snaps_wca):
        col = cw(i / max(ns2-1, 1))
        lbl = f"t={t:.0f}τ" if i in [0, ns2//2, ns2-1] else ""
        ax_sw.plot(k, S, color=col, lw=1.6, alpha=0.9, label=lbl)
        i_pk = np.argmax(S)
        ax_sw.plot(k[i_pk], S[i_pk], 'o', color=col, ms=4)
    ax_sw.set_title("S(k) — SANS GEL / WCA  (○=pic k*)", **tkw)
    ax_sw.set_xlabel("k [σ⁻¹]", color="#888", fontsize=8)
    ax_sw.set_ylabel("S(k)", color="#888", fontsize=8)
    ax_sw.tick_params(colors="#555", labelsize=7)
    ax_sw.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
    for sp in ax_sw.spines.values(): sp.set_edgecolor("#222")

    ax_sc.set_facecolor("#0d0d0d")
    tg, kg, Sg = snaps_gel[-1]
    tw, kw2, Sw = snaps_wca[-1]
    ax_sc.plot(kg,  Sg, color="#ff6b35", lw=2.2, label=f"Gel  t={tg:.0f}τ")
    ax_sc.plot(kw2, Sw, color="#66ddff", lw=2.2, ls="--", label=f"WCA  t={tw:.0f}τ")
    ax_sc.set_title("S(k) final : GEL vs SANS GEL", **tkw)
    ax_sc.set_xlabel("k [σ⁻¹]", color="#888", fontsize=8)
    ax_sc.set_ylabel("S(k)", color="#888", fontsize=8)
    ax_sc.tick_params(colors="#555", labelsize=7)
    ax_sc.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", framealpha=0.7)
    for sp in ax_sc.spines.values(): sp.set_edgecolor("#222")

    # ── Ligne 2 : k*(t), S(k*,t), positions finales ──
    ax_kstar = fig.add_subplot(gs[2, 0])
    ax_sstar = fig.add_subplot(gs[2, 1])
    ax_pf    = fig.add_subplot(gs[2, 2])

    t_gel  = [s[0] for s in snaps_gel]
    kstar_gel = [s[1][np.argmax(s[2])] for s in snaps_gel]
    Sstar_gel = [np.max(s[2]) for s in snaps_gel]
    t_wca  = [s[0] for s in snaps_wca]
    kstar_wca = [s[1][np.argmax(s[2])] for s in snaps_wca]
    Sstar_wca = [np.max(s[2]) for s in snaps_wca]

    ax_kstar.set_facecolor("#0d0d0d")
    ax_kstar.plot(t_gel, kstar_gel, 'o-', color="#ff6b35", lw=1.8, ms=5, label="k* gel")
    ax_kstar.plot(t_wca, kstar_wca, 's--', color="#66ddff", lw=1.8, ms=5, label="k* WCA")
    ax_kstar.set_title("k*(t) — arrêt = k* constant", **tkw)
    ax_kstar.set_xlabel("t [τ]", color="#888", fontsize=8)
    ax_kstar.set_ylabel("k* [σ⁻¹]", color="#888", fontsize=8)
    ax_kstar.tick_params(colors="#555", labelsize=7)
    ax_kstar.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
    for sp in ax_kstar.spines.values(): sp.set_edgecolor("#222")

    ax_sstar.set_facecolor("#0d0d0d")
    ax_sstar.plot(t_gel, Sstar_gel, 'o-', color="#ff6b35", lw=1.8, ms=5, label="S(k*) gel")
    ax_sstar.plot(t_wca, Sstar_wca, 's--', color="#66ddff", lw=1.8, ms=5, label="S(k*) WCA")
    ax_sstar.set_title("S(k*,t) — croissance = coarsening", **tkw)
    ax_sstar.set_xlabel("t [τ]", color="#888", fontsize=8)
    ax_sstar.set_ylabel("S(k*)", color="#888", fontsize=8)
    ax_sstar.tick_params(colors="#555", labelsize=7)
    ax_sstar.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
    for sp in ax_sstar.spines.values(): sp.set_edgecolor("#222")

    # Positions finales comparées
    ax_pf.set_facecolor("#0d0d0d")
    ax_pf.scatter(pos_gel[:,0], pos_gel[:,1],
                  c="#ff6b35", s=7, alpha=0.8, linewidths=0, label="Gel")
    ax_pf.scatter(pos_wca[:,0]+L*0.0, pos_wca[:,1],
                  c="#66ddff", s=7, alpha=0.4, linewidths=0, label="WCA")
    ax_pf.set_xlim(0, L); ax_pf.set_ylim(0, L); ax_pf.set_aspect("equal")
    ax_pf.set_title("Positions finales — GEL (orange) vs WCA (bleu)", **tkw)
    ax_pf.tick_params(colors="#555", labelsize=7)
    for sp in ax_pf.spines.values(): sp.set_edgecolor("#222")

    fig.suptitle(
        f"Dynamique Brownienne — Coarsening & Gélation Colloïdale  [v6]\n"
        f"N={N}  φ={PHI}  dt={DT}  ε_gel={EPS_GEL:.0f}kBT"
        f"  t_final={N_STEPS*DT:.0f}τ  |  force_cap={FORCE_CAP}  (Euler-Maruyama + numba)",
        color="white", fontsize=9.5, fontfamily="monospace", y=0.97
    )
    out = "colloidal_gel_v6.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Figure sauvegardée : {out}")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    t_final = N_STEPS * DT
    print("="*60)
    print(f"  N={N}, {N_STEPS} pas, dt={DT}, t_final={t_final:.0f}τ")
    print(f"  ε_gel={EPS_GEL}kBT  force_cap={FORCE_CAP}  numba @njit")
    print(f"  ~30-60 min total selon CPU")
    print("="*60)

    snaps_gel, en_gel, pos_gel, L, ps_gel = run_simulation(mode="gel")
    snaps_wca, en_wca, pos_wca, _, ps_wca = run_simulation(mode="no_gel")
    plot_results(snaps_gel, snaps_wca, pos_gel, pos_wca, L,
                 en_gel, en_wca, ps_gel, ps_wca)
