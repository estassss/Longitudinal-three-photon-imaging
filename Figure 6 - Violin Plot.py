import os
import numpy as np
import scipy.io as sio
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# ------------------------------- Constants -----------------------------------
Z_SPACING = 5.0  # µm 
XY_PIXEL_SIZE_UM = 500.0 / 512.0

OUTPUT_DIR = "/Users/elinestas/Desktop/Box_Plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- Session + Age ---------------------------
session_to_age = {
    'Mouse 1 FOV I':   {'Session 1': 8.9, 'Session 2': 9.9},
    'Mouse 1 FOV II':  {'Session 1': 8.9, 'Session 2': 9.9},
    'Mouse 2 FOV II':  {'Session 1': 6.5, 'Session 2': 7.5},
    'Mouse 3 FOV I':   {'Session 1': 6.5, 'Session 2': 7.5},
}

# ----------------------------- File paths (Vessels) --------------------------
file_paths_v = {
    'Mouse 1 FOV I': {
        'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.1.V-V_fwd.mat',
        'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.2.V-V_fwd.mat',
    },
    'Mouse 1 FOV II': {
        'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.1.S-V_fwd.mat',
        'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.2.S-V_fwd.mat',
    },
    'Mouse 2 FOV II': {
        'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-223783.1.S-V_fwd.mat',
        'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-223783.2.S-V_fwd.mat',
    },
    'Mouse 3 FOV I': {
        'Session 1': '/Users/elinestas/Desktop/untitled folder/Vess_Architecture-Analysis-Mouse3FOVI_Session1-V_fwd.mat',
        'Session 2': '/Users/elinestas/Desktop/untitled folder/Vess_Architecture-Analysis-Mouse3FOVI_Session2-V_fwd.mat',
    },
}

# ----------------------------- File paths (Plaques) --------------------------
file_paths_p = {
    'Mouse 1 FOV I': {
        'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.1.V.npy',
        'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.2.V.npy',
    },
    'Mouse 1 FOV II': {
        'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.1.S.npy',
        'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.2.S.npy',
    },
    'Mouse 2 FOV II': {
        'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223783.1.S.npy',
        'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223783.2.S.npy',
    },
    'Mouse 3 FOV I': {
        'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223785.1.V.npy',
        'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223785.2.V.npy',
    },
}

color_palette = {
    'Mouse 1 FOV I':  '#2E77BB',  # blue
    'Mouse 1 FOV II': '#C5662D',  # burnt orange
    'Mouse 2 FOV II': '#2FA07D',  # teal green
    'Mouse 3 FOV I':  '#E7DC57',  # soft yellow
}

# Short x-tick labels
display_label = {
    'Mouse 1 FOV I':  'M1F1',
    'Mouse 1 FOV II': 'M1F2',
    'Mouse 2 FOV II': 'M2F2',
    'Mouse 3 FOV I':  'M3F1',
}

# ----------------------------- Parameters to analyze -------------------------
parameters = [
    "Vessel Diameter",
    "Vessel Length",
    "Vessel Tortuosity",
    "Inter-Vessel Distance",
    "Plaque Radius",
]

# ------------------------------ Stats helpers --------------------------------
def is_normal_distribution(data, alpha=0.05):
    if len(data) < 3:
        return True
    stat, p_value = shapiro(data)
    return p_value > alpha

def compare_groups(s1, s2):
    if len(s1) < 2 or len(s2) < 2:
        return None, None
    normal_s1 = is_normal_distribution(s1)
    normal_s2 = is_normal_distribution(s2)
    if normal_s1 and normal_s2:
        stat, p_val = ttest_ind(s1, s2, equal_var=False)
        return p_val, "Welch t-test"
    stat, p_val = mannwhitneyu(s1, s2, alternative='two-sided')
    return p_val, "Mann-Whitney U"

# ------------------------------- Loaders -------------------------------------
def load_data_p(file_path):
    if file_path and os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True).item()
    return None

def load_data_v(file_path):
    if file_path and os.path.exists(file_path):
        return sio.loadmat(file_path, simplify_cells=True)
    return None

def _order_skeleton_longest_path(coords_yxz: np.ndarray) -> np.ndarray:
    import numpy as _np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    n = coords_yxz.shape[0]
    if n < 3:
        return coords_yxz
    coords = coords_yxz.astype(int)
    lookup = {tuple(c): i for i, c in enumerate(coords)}
    rows, cols = [], []
    for i, (y, x, z) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dy == dx == dz == 0:
                        continue
                    j = lookup.get((y+dy, x+dx, z+dz))
                    if j is not None:
                        rows.append(i); cols.append(j)
    if not rows:
        return coords_yxz
    G = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    d0 = shortest_path(G, directed=False, unweighted=True, indices=0)
    a = int(np.nanargmax(d0))
    da = shortest_path(G, directed=False, unweighted=True, indices=a)
    b = int(np.nanargmax(da))
    _, pred = shortest_path(G, directed=False, unweighted=True, indices=a, return_predecessors=True)
    if pred[b] == -9999:
        return coords_yxz
    path = []
    cur = b
    while cur != -9999 and cur != a:
        path.append(cur)
        cur = int(pred[cur])
    path.append(a)
    path.reverse()
    return coords[path, :]

def _metrics_from_skel_fallback(data_mat, xy_um=XY_PIXEL_SIZE_UM, z_um=Z_SPACING, anisotropic=False):
    import numpy as _np
    from scipy import ndimage as _ndi

    V = data_mat.get('V1', data_mat.get('V', None))
    if V is None:
        return {k: _np.array([]) for k in ('Vessel Diameter','Vessel Length','Vessel Tortuosity','Inter-Vessel Distance')}
    V = _np.asarray(V).astype(bool)
    if V.ndim != 3:
        return {k: _np.array([]) for k in ('Vessel Diameter','Vessel Length','Vessel Tortuosity','Inter-Vessel Distance')}
    if 'Skel' not in data_mat:
        return {k: _np.array([]) for k in ('Vessel Diameter','Vessel Length','Vessel Tortuosity','Inter-Vessel Distance')}

    Skel_cell = _np.asarray(data_mat['Skel']).ravel()
    dist = _ndi.distance_transform_edt(V)  # foreground EDT (voxels)

    N = Skel_cell.size
    diam = _np.zeros(N, float)
    leng = _np.zeros(N, float)
    tort = _np.ones(N, float)
    mids = _np.zeros((N,3), float) * _np.nan

    for i, seg in enumerate(Skel_cell):
        seg = _np.asarray(seg)
        if seg.size == 0:
            diam[i] = _np.nan; leng[i] = 0.0; tort[i] = 1.0
            continue
        # [x y z] 1-based → [y,x,z] 0-based
        yxz = _np.stack([seg[:,1]-1, seg[:,0]-1, seg[:,2]-1], axis=1).astype(int)
        yxz[:,0] = _np.clip(yxz[:,0], 0, V.shape[0]-1)
        yxz[:,1] = _np.clip(yxz[:,1], 0, V.shape[1]-1)
        yxz[:,2] = _np.clip(yxz[:,2], 0, V.shape[2]-1)

        # Diameter (voxel) → µm
        r = dist[yxz[:,0], yxz[:,1], yxz[:,2]]
        diam[i] = float(np.nanmedian(2.0*r)) * xy_um if r.size else np.nan

        # Length/Tortuosity (assume XY spacing only for fallback)
        yxz_ord = _order_skeleton_longest_path(yxz)
        if yxz_ord.shape[0] > 1:
            diffs = np.diff(yxz_ord.astype(float), axis=0) * np.array([xy_um, xy_um, 0.0])
            seg_len_um = np.sqrt((diffs**2).sum(axis=1)).sum()
            leng[i] = float(seg_len_um)
            delta_um = (yxz_ord[0].astype(float) - yxz_ord[-1].astype(float)) * np.array([xy_um, xy_um, 0.0])
            straight_um = float(np.linalg.norm(delta_um))
            tort[i] = float(leng[i] / straight_um) if straight_um > 0 else 1.0
        else:
            leng[i] = 0.0; tort[i] = 1.0
        mids[i] = yxz_ord.mean(axis=0)

    # Inter-vessel distance via nearest midpoint (XY only here)
    if N >= 2 and np.all(np.isfinite(mids)):
        d_vox = np.sqrt(((mids[:,None,:] - mids[None,:,:])**2).sum(axis=2))
        np.fill_diagonal(d_vox, np.inf)
        ivd = (d_vox.min(axis=1)) * xy_um
    else:
        ivd = np.array([])

    return {
        'Vessel Diameter':       diam,
        'Vessel Length':         leng,
        'Vessel Tortuosity':     tort,
        'Inter-Vessel Distance': ivd,
    }

def extract_parameters_v(data):
    # Try precomputed arrays first
    diam = data.get('Diameters', None)
    leng = data.get('Lengths', None)
    tort = data.get('Tortuosities', data.get('Tortuocities', None))
    ivds = data.get('InterVesselDistances', None)

    def _bad(arr):
        a = np.asarray(arr) if arr is not None else None
        if a is None or a.size == 0:
            return True
        return np.all(a == 0) or np.all(~np.isfinite(a))

    if not (_bad(diam) or _bad(leng) or _bad(tort) or _bad(ivds)):
        return {
            'Vessel Diameter':       np.abs(np.asarray(diam).ravel()) * XY_PIXEL_SIZE_UM,
            'Vessel Length':         np.abs(np.asarray(leng).ravel()) * XY_PIXEL_SIZE_UM,
            'Vessel Tortuosity':     np.abs(np.asarray(tort).ravel()),
            'Inter-Vessel Distance': np.abs(np.asarray(ivds).ravel()) * XY_PIXEL_SIZE_UM,
        }

    # Fallback: compute from V1+Skel
    return _metrics_from_skel_fallback(data, xy_um=XY_PIXEL_SIZE_UM, z_um=Z_SPACING, anisotropic=False)

def extract_parameters_p(plaque_data):
    if 'plaque_parameters' not in plaque_data:
        return {'Plaque Radius': np.array([])}
    pp = plaque_data['plaque_parameters']
    if isinstance(pp, dict):
        vols = np.asarray(pp.get('Volume (um^3)', []), dtype=float).ravel()
    else:
        arr = np.asarray(pp)
        if getattr(arr, 'dtype', None) is not None and arr.dtype.names:
            for k in ('Volume (um^3)', 'Volume_um3', 'Volume', 'volume', 'vol'):
                if k in arr.dtype.names:
                    vols = np.asarray(arr[k], dtype=float).ravel()
                    break
            else:
                vols = arr.astype(float).ravel()
        else:
            vols = arr.astype(float).ravel()
    if vols.size == 0:
        return {'Plaque Radius': np.array([])}
    radius = (3.0 * np.abs(vols) / (4.0 * np.pi)) ** (1.0 / 3.0)
    return {'Plaque Radius': radius}

def load_and_process_metrics():
    metrics_by_mouse = {}
    mice = sorted(set(file_paths_v.keys()) | set(file_paths_p.keys()))
    for mouse in mice:
        metrics_by_mouse[mouse] = {}
        sessions = set(file_paths_v.get(mouse, {}).keys()) | set(file_paths_p.get(mouse, {}).keys())
        for session in sorted(sessions):
            combined = {}
            v_path = file_paths_v.get(mouse, {}).get(session, None)
            p_path = file_paths_p.get(mouse, {}).get(session, None)
            if v_path:
                vdat = load_data_v(v_path)
                if vdat:
                    combined.update(extract_parameters_v(vdat))
            if p_path:
                pdat = load_data_p(p_path)
                if pdat:
                    combined.update(extract_parameters_p(pdat))
            if any(len(v) > 0 for v in combined.values()):
                metrics_by_mouse[mouse][session] = combined
    return metrics_by_mouse

def extract_data_for_boxplots(metrics_by_mouse):
    mice = sorted(metrics_by_mouse.keys())
    sessions = ["Session 1", "Session 2"]
    data = {param: {m: {s: [] for s in sessions} for m in mice} for param in parameters}
    for mouse, sess_dict in metrics_by_mouse.items():
        for session, metrics in sess_dict.items():
            for param in parameters:
                if param in metrics:
                    data[param][mouse][session] = list(np.asarray(metrics[param]).astype(float))
    return data

# ------------------------------- Plotting ------------------------------------
def plot_all_violins(data,
                     color_palette=color_palette,
                     parameters=parameters,
                     OUTPUT_DIR=OUTPUT_DIR,
                     age_dict=session_to_age):

    def lighter(c, f=0.55):
        r, g, b = mcolors.to_rgb(c)
        return tuple(1 - (1 - x) * f for x in (r, g, b))

    def p_to_stars(p):
        return "***" if p is not None and p < 0.001 else ("**" if p is not None and p < 0.01 else ("*" if p is not None and p < 0.05 else ""))

    mice = sorted(data[parameters[0]].keys())
    session_names = ["Session 1", "Session 2"]
    order = [f"{m}_{s}" for m in mice for s in session_names]

    palette = {}
    for m in mice:
        base = color_palette.get(m, 'gray')
        palette[f"{m}_Session 1"] = base
        palette[f"{m}_Session 2"] = lighter(base)

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    gs  = GridSpec(2, 3, figure=fig, hspace=.55, wspace=.45)
    axes  = [fig.add_subplot(gs[i]) for i in range(5)]
    lg_ax = fig.add_subplot(gs[1, 2]);  lg_ax.axis("off")

    log_y = {"Vessel Length", "Vessel Tortuosity"}

    for ax, param, letter in zip(axes, parameters[:5], ["(a)", "(b)", "(c)", "(d)", "(e)"]):
        rows = []
        for m in mice:
            for s in session_names:
                vals = data.get(param, {}).get(m, {}).get(s, [])
                rows += [{"Value": float(v), "Mouse": m, "Session": s, "Cat": f"{m}_{s}"} for v in vals]
        df = pd.DataFrame(rows)

        sns.violinplot(ax=ax, data=df, x="Cat", y="Value",
                       order=order, palette=palette,
                       bw=.15, cut=0, width=.30, linewidth=.8,
                       saturation=1, inner="quartile")

        # whisker + median line per category
        for i, cat in enumerate(order):
            vals = df.loc[df.Cat == cat, "Value"].to_numpy()
            if vals.size == 0:
                continue
            vmin, vmax = vals.min(), vals.max()
            med = np.median(vals)
            ax.plot([i, i], [vmin, vmax], color="black", lw=.8, zorder=3)
            ax.plot([i-.15, i+.15], [med, med], color="black", lw=2.2, zorder=4)

        if param in log_y:
            pos = df.Value[df.Value > 0]
            if not pos.empty:
                ax.set_yscale("log")
                ax.set_ylim(bottom=pos.min()*0.5)

        centres = [0.5 + 2*i for i in range(len(mice))]
        ax.set_xticks(centres)
        ax.set_xticklabels([display_label.get(m, m) for m in mice],
                           fontsize=10, weight="bold")

        y_unit = "µm" if param != "Vessel Tortuosity" else ""
        ax.set_ylabel(f"{param} {f'({y_unit})' if y_unit else ''}".strip(), fontsize=11, weight="bold")
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
                fontsize=14, weight="bold", va="center")

        top = ax.get_ylim()[1]
        # within-mouse S1 vs S2 test
        for m_idx, m in enumerate(mice):
            s1 = df[(df.Mouse == m) & (df.Session == "Session 1")]["Value"].to_numpy()
            s2 = df[(df.Mouse == m) & (df.Session == "Session 2")]["Value"].to_numpy()
            p, _test = compare_groups(s1, s2)
            stars = p_to_stars(p)
            if stars:
                x1, x2 = 2*m_idx, 2*m_idx + 1
                y = top*0.95 if param not in log_y else top/1.1
                ax.plot([x1, x2], [y, y], color="black", lw=1.2)
                ax.text((x1+x2)/2, y*1.02, stars, ha="center", va="bottom",
                        fontsize=12, weight="bold")

    # Legend with short labels
    legend_handles = []
    for m in mice:
        base = color_palette.get(m, 'gray')
        age1 = age_dict.get(m, {}).get("Session 1", "S1")
        age2 = age_dict.get(m, {}).get("Session 2", "S2")
        lighter_col = tuple(1 - (1 - x) * 0.55 for x in mcolors.to_rgb(base))
        legend_handles.append(Patch(color=base,        label=f"{display_label.get(m,m)}: {age1} mo (S1)"))
        legend_handles.append(Patch(color=lighter_col, label=f"{display_label.get(m,m)}: {age2} mo (S2)"))

    lg_ax.legend(handles=legend_handles,
                 title="Age by Mouse/FOV (months)",
                 frameon=False, fontsize=10, title_fontsize=11,
                 loc="upper left")

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "All_Violins.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Violin panels saved → {out}")

# ----------------------------------- Main ------------------------------------
def main():
    metrics = load_and_process_metrics()
    data_for_boxplots = extract_data_for_boxplots(metrics)
    plot_all_violins(data_for_boxplots)

if __name__ == "__main__":
    main()
