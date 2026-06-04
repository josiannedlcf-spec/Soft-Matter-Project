"""
Minimal Brownian dynamics for attractive colloids in 2D.
Two regimes are provided:
  - free_coarsening: weaker attraction, mobile aggregates
  - gel_arrest: strong short-range attraction + local-mobility slowdown to mimic arrest
Outputs: snapshots, S(q,t), length ell(t), MSD, percolation flag.

Run: python brownian_gel_bd.py
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ---------------- Parameters ----------------
N = 400                 # particles; increase to 800 if your PC is OK
phi = 0.18              # 2D area fraction: N*pi*(sigma/2)^2/L^2
sigma = 1.0             # particle diameter = length unit
D0 = 1.0                # Brownian diffusion coefficient = time unit sigma^2/D0
kBT = 1.0
alpha = 18.0            # attraction inverse range; high = short range
r0 = sigma
rc = r0 + 5.0/alpha
bond_cut = 1.15*sigma

dt = 2e-5               # small because Morse forces can be stiff
nsteps = 120000
save_every = 3000
seed = 7

# Choose "free_coarsening" or "gel_arrest"
case = "gel_arrest"

if case == "free_coarsening":
    eps = 2.0           # moderate attraction
    arrest_slowdown = False
elif case == "gel_arrest":
    eps = 6.0           # deep quench: strong attraction
    arrest_slowdown = True
else:
    raise ValueError("case must be free_coarsening or gel_arrest")

rng = np.random.default_rng(seed)
L = np.sqrt(N*np.pi*(sigma/2)**2/phi)

# q-vectors for isotropic structure factor
nmax = 18
qvecs = []
for nx in range(-nmax, nmax+1):
    for ny in range(-nmax, nmax+1):
        if nx == 0 and ny == 0:
            continue
        qvecs.append(2*np.pi*np.array([nx, ny])/L)
qvecs = np.array(qvecs)
qmag = np.linalg.norm(qvecs, axis=1)
q_bins = np.linspace(2*np.pi/L, np.max(qmag), 55)
q_centers = 0.5*(q_bins[:-1] + q_bins[1:])

# ---------------- Utilities ----------------
def pbc(x):
    return x % L

def min_image(dr):
    return dr - L*np.rint(dr/L)

def init_lattice_noise():
    """Initial condition without overlaps."""
    nside = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0, L, nside, endpoint=False)
    grid = np.array([(x, y) for x in xs for y in xs])[:N]
    pos = grid + 0.15*sigma*rng.normal(size=(N, 2))
    return pbc(pos)

def forces_and_bonds(pos):
    """Morse forces, coordination, and bond adjacency. O(N^2), OK for N~400."""
    F = np.zeros_like(pos)
    z = np.zeros(N, dtype=int)
    adj = [[] for _ in range(N)]
    for i in range(N-1):
        dr = min_image(pos[i+1:] - pos[i])
        r = np.linalg.norm(dr, axis=1)
        mask = r < rc
        if np.any(mask):
            idx = np.where(mask)[0] + i + 1
            rr = r[mask]
            drr = dr[mask]
            exp1 = np.exp(-alpha*(rr-r0))
            # F_i due to j: 2 alpha eps (exp(-2a...) - exp(-a...)) r_ij/r
            scalar = 2*alpha*eps*(exp1**2 - exp1)/rr
            fij = scalar[:, None]*drr
            F[i] += np.sum(fij, axis=0)
            F[idx] -= fij
        bmask = r < bond_cut
        if np.any(bmask):
            idxb = np.where(bmask)[0] + i + 1
            z[i] += len(idxb)
            z[idxb] += 1
            for j in idxb:
                adj[i].append(j)
                adj[j].append(i)
    return F, z, adj

def local_mobility(z):
    """Phenomenological arrest: dense highly bonded regions diffuse much less."""
    if not arrest_slowdown:
        return np.ones_like(z, dtype=float)
    # z~4 or larger becomes almost immobile; tune if needed
    return 1.0/(1.0 + np.exp(2.2*(z-3.5))) + 0.015

def heun_step(pos):
    F0, z0, _ = forces_and_bonds(pos)
    m0 = local_mobility(z0)[:, None]
    noise = np.sqrt(2*D0*m0*dt)*rng.normal(size=pos.shape)
    pred = pbc(pos + m0*F0*dt + noise)
    F1, z1, _ = forces_and_bonds(pred)
    m1 = local_mobility(z1)[:, None]
    # Same noise in predictor and corrector; deterministic drift averaged.
    return pbc(pos + 0.5*(m0*F0 + m1*F1)*dt + noise)

def structure_factor(pos):
    phase = qvecs @ pos.T
    rho = np.exp(1j*phase).sum(axis=1)
    Sq_raw = (np.abs(rho)**2)/N
    Sq = np.zeros(len(q_centers))
    counts = np.zeros(len(q_centers))
    inds = np.digitize(qmag, q_bins) - 1
    for val, ind in zip(Sq_raw, inds):
        if 0 <= ind < len(Sq):
            Sq[ind] += val
            counts[ind] += 1
    Sq = np.divide(Sq, counts, out=np.full_like(Sq, np.nan), where=counts > 0)
    return q_centers, Sq

def characteristic_length(q, Sq):
    valid = np.isfinite(Sq)
    # ignore the very first bin, often noisy due to finite size
    qq, ss = q[valid][2:], Sq[valid][2:]
    imax = np.argmax(ss)
    return 2*np.pi/qq[imax], qq[imax], ss[imax]

def percolates(pos, adj):
    """Simple visual/percolation criterion: bonded cluster spans > 0.75 L in x or y."""
    seen = np.zeros(N, dtype=bool)
    for s in range(N):
        if seen[s]:
            continue
        comp = []
        dq = deque([s]); seen[s] = True
        while dq:
            i = dq.popleft(); comp.append(i)
            for j in adj[i]:
                if not seen[j]:
                    seen[j] = True; dq.append(j)
        if len(comp) > 5:
            pts = pos[comp]
            if pts[:,0].ptp() > 0.75*L or pts[:,1].ptp() > 0.75*L:
                return True, len(comp)
    return False, 0

def plot_snapshot(pos, step):
    plt.figure(figsize=(5,5))
    plt.scatter(pos[:,0], pos[:,1], s=8)
    plt.xlim(0,L); plt.ylim(0,L); plt.gca().set_aspect('equal')
    plt.title(f"{case}, t={step*dt:.2f}")
    plt.tight_layout()
    plt.savefig(f"snapshot_{case}_{step:07d}.png", dpi=160)
    plt.close()

filename = f"snapshot_{case}_{step:07d}.png"
plt.savefig(filename, dpi=160)
print("snapshot saved:", filename)

# ---------------- Run ----------------
pos = init_lattice_noise()
pos0 = pos.copy()
times, ells, qstars, smaxs, msds, percs = [], [], [], [], [], []
S_list = []

for step in range(nsteps+1):
    if step % save_every == 0:
        F, z, adj = forces_and_bonds(pos)
        q, Sq = structure_factor(pos)
        ell, qstar, smax = characteristic_length(q, Sq)
        dr = min_image(pos - pos0)
        msd = np.mean(np.sum(dr*dr, axis=1))
        perc, csize = percolates(pos, adj)

        times.append(step*dt); ells.append(ell); qstars.append(qstar)
        smaxs.append(smax); msds.append(msd); percs.append(perc)
        S_list.append(Sq)
        print(f"step {step:7d} t={step*dt:7.3f} ell={ell:6.2f} q*={qstar:6.3f} Smax={smax:7.2f} MSD={msd:7.3f} perc={perc}")
        if step in [0, nsteps//4, nsteps//2, nsteps]:
            plot_snapshot(pos, step)
    if step < nsteps:
        pos = heun_step(pos)

# ---------------- Plots ----------------
S_arr = np.array(S_list)
times = np.array(times); ells = np.array(ells); msds = np.array(msds); percs = np.array(percs)

plt.figure(figsize=(6,4))
for i in np.linspace(0, len(times)-1, 6, dtype=int):
    plt.plot(q, S_arr[i], label=f"t={times[i]:.2f}")
plt.xlabel("q")
plt.ylabel("S(q,t)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"Sq_{case}.png", dpi=180)

plt.figure(figsize=(6,4))
plt.loglog(times[1:], ells[1:], "o-", label="simulation")
plt.loglog(times[1:], ells[1]/(times[1]**(1/3))*times[1:]**(1/3), "--", label=r"$t^{1/3}$ guide")
plt.xlabel("t")
plt.ylabel(r"$\ell(t)=2\pi/q^*$")
plt.legend()
plt.tight_layout()
plt.savefig(f"length_{case}.png", dpi=180)

plt.figure(figsize=(6,4))
plt.loglog(times[1:], msds[1:], "o-")
plt.xlabel("t")
plt.ylabel("MSD")
plt.tight_layout()
plt.savefig(f"MSD_{case}.png", dpi=180)

plt.figure(figsize=(6,4))
for i in np.linspace(max(1, len(times)//4), len(times)-1, 5, dtype=int):
    plt.plot(q*ells[i], S_arr[i]/ells[i]**2, label=f"t={times[i]:.2f}")
plt.xlabel(r"$q\ell(t)$")
plt.ylabel(r"$S(q,t)/\ell(t)^2$")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"scaling_{case}.png", dpi=180)

np.savez(f"data_{case}.npz", q=q, S=S_arr, times=times, ell=ells, msd=msds, percolates=percs)
print("Done. Figures saved as PNG and data saved as NPZ.")
