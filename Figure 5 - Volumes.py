import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

# Constants (µm units)
Z_SPACING_UM = 5.0
XY_PIXEL_SIZE_UM = 500.0 / 512.0
VOXEL_UM3 = (XY_PIXEL_SIZE_UM ** 2) * Z_SPACING_UM  # 5.24288 µm^3

# File paths
file_paths_v = {
    'Mouse 1': {
        'I': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.1.V-V_fwd.mat',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.2.V-V_fwd.mat',
        },
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.1.S-V_fwd.mat',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-58283.2.S-V_fwd.mat',
        },
    },
    'Mouse 2': {
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-223783.1.S-V_fwd.mat',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-223783.2.S-V_fwd.mat',
        },
    },
    'Mouse 3': {
        'I': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-from-Mouse3FOVI_Session1.1.mat',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Vessels/Vess_Architecture-Analysis-from-Mouse3FOVI_Session2.1.mat',
        },
    },
}

file_paths_p = {
    'Mouse 1': {
        'I': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.1.V.npy',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.2.V.npy',
        },
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.1.S.npy',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/58283.2.S.npy',
        },
    },
    'Mouse 2': {
        'II': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223783.1.S.npy',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223783.2.S.npy',
        },
    },
    'Mouse 3': {
        'I': {
            'Session 1': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223785.1.V.npy',
            'Session 2': '/Users/elinestas/Desktop/VAβs/Data/4. Segmented/Plaques/223785.2.V.npy',
        },
    },
}

# Order
SET_ORDER = [
    "Mouse 1 — I",
    "Mouse 1 — II",
    "Mouse 2 — I",   # change to "Mouse 2 — II" if that’s your real dataset label
    "Mouse 3 — I",
]

# COlors
SET_COLORS = {
    "Mouse 1 — I":  "#1f77b4",  # blue
    "Mouse 1 — II": "#ff7f0e",  # orange
    "Mouse 2 — I":  "#2ca02c",  # green
    "Mouse 3 — I":  "#f0e442",  # yellow
}

# Functions load data
def load_data_p(path):
    if not os.path.exists(path):
        return None
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception:
        return None

def load_data_v(path):
    if not os.path.exists(path):
        return None
    try:
        return sio.loadmat(path)
    except Exception:
        return None

# Extracting parameters plaques
def extract_parameters_p(plaque_data):
    if not isinstance(plaque_data, dict):
        return {'volume_um3': np.array([])}
    if 'plaque_parameters' not in plaque_data:
        return {'volume_um3': np.array([])}
    try:
        vol = np.abs(plaque_data['plaque_parameters']['Volume (um^3)']).astype(float).ravel()
        return {'volume_um3': vol}
    except Exception:
        return {'volume_um3': np.array([])}

# Find correct columns in DeepVess .mat file
def _find_vessel_stack(mat):
    keys = ['V1','V','Vessels','VesselStack','Stack','BW','BWstack','Binary','binary','mask','seg']
    for k in keys:
        if k in mat:
            arr = np.asarray(mat[k])
            if arr.ndim >= 3:
                return arr
    for v in mat.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v:
                    arr = np.asarray(v[k])
                    if arr.ndim >= 3:
                        return arr
        if isinstance(v, np.ndarray) and v.dtype == object:
            for it in v.ravel():
                try:
                    arr = np.asarray(it)
                    if arr.ndim >= 3:
                        return arr
                except Exception:
                    pass
    return None

# Determine volume using voxel size
def vessel_volume_from_nnz_um3(mat):
    stack = _find_vessel_stack(mat)
    if stack is None:
        return np.nan
    return int(np.count_nonzero(stack)) * VOXEL_UM3

# Load metrics per area
def load_area_metrics(area):
    rows = []
    for mouse, areas in file_paths_p.items():
        if area not in areas: 
            continue
        if mouse not in file_paths_v or area not in file_paths_v[mouse]:
            continue
        for session, p_path in areas[area].items():
            v_path = file_paths_v[mouse][area].get(session)
            if not v_path:
                continue
            p = load_data_p(p_path)
            v = load_data_v(v_path)
            if p is None or v is None:
                continue

            plaque_vol = float(np.sum(extract_parameters_p(p)['volume_um3']))
            vessel_vol = float(vessel_volume_from_nnz_um3(v))

            rows.append({
                "mouse": mouse,
                "area": area,
                "session": session,
                "plaque_vol_um3": plaque_vol,
                "vessel_vol_um3": vessel_vol,
            })
    return rows

def load_all_rows():
    all_rows = []
    for area in ('I','II'):
        all_rows.extend(load_area_metrics(area))
    return all_rows

# Determine which session it is (1 vs 2)
def _collect_session_vals(all_rows, metric_key):
    from collections import defaultdict
    bucket = defaultdict(lambda: {"Session 1": [], "Session 2": []})
    for r in all_rows:
        set_key = f"{r['mouse']} — {r['area']}"
        if metric_key in r and np.isfinite(r[metric_key]):
            bucket[set_key][r['session']].append(float(r[metric_key]))
    # take means per session
    out = {}
    for s, dd in bucket.items():
        s1 = np.nanmean(dd["Session 1"]) if len(dd["Session 1"]) else np.nan
        s2 = np.nanmean(dd["Session 2"]) if len(dd["Session 2"]) else np.nan
        out[s] = (s1, s2)
    return out

# Compute percentage difference S1 vs S2
def compute_pct_df(all_rows, metric_key):
    sess = _collect_session_vals(all_rows, metric_key)
    recs = []
    for s in SET_ORDER:
        s1, s2 = sess.get(s, (np.nan, np.nan))
        pct = 100.0 * (s2 - s1) / s1 if np.isfinite(s1) and s1 != 0 and np.isfinite(s2) else np.nan
        recs.append({"Set": s, "S1": s1, "S2": s2, "PctChange": pct})
    return pd.DataFrame(recs)

# Plot as barchart
def bar_percent_fixed(df, title, ylabel, outfile, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    x_labels = df["Set"].tolist()
    pcts = df["PctChange"].to_numpy()
    colors = [SET_COLORS.get(s, "#999999") for s in x_labels]

    x = np.arange(len(x_labels))
    plt.figure(figsize=(8.5, 4.8), facecolor="white")

    # 4 bars and NaN if file not found
    plot_vals = np.where(np.isfinite(pcts), pcts, 0.0)
    bars = plt.bar(x, plot_vals, color=colors, edgecolor='black', linewidth=0.8)

    # Zero line
    plt.axhline(0, color='black', linewidth=0.9)

    # Labels
    xtick_labels = []
    for s in x_labels:
        mouse, area = s.split(' — ')
        roman_mouse = mouse.replace("Mouse ", "Mouse ")
        xtick_labels.append(f"{roman_mouse}\nFOV {area}")
    plt.xticks(x, xtick_labels, ha='center')

    # Annotate values
    ymax = np.nanmax(np.abs(plot_vals)) if np.isfinite(plot_vals).any() else 1.0
    offset = 0.02 * (ymax if ymax > 0 else 1.0)
    for xi, pct, bar in zip(x, pcts, bars):
        if np.isfinite(pct):
            txt = f"{pct:+.1f}%"
            va = 'bottom' if pct >= 0 else 'top'
            y = bar.get_height()
            plt.text(xi, y + (offset if y >= 0 else -offset), txt, ha='center', va=va, fontsize=10)
        else:
            plt.text(xi, 0 + offset, "(no data)", ha='center', va='bottom', fontsize=8)

    plt.ylabel(ylabel)
    plt.title(title, weight="bold")
    plt.tight_layout()

    out = os.path.join(output_folder, outfile)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Saved: {out}")

def plot_percent_change_barcharts(all_rows, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Vessel
    df_v = compute_pct_df(all_rows, "vessel_vol_um3")
    df_v.to_csv(os.path.join(output_folder, "PctChange_Vessels_by_Set.csv"), index=False)
    bar_percent_fixed(
        df_v,
        title="Δ% Vessels (Session 2 vs Session 1)",
        ylabel="Δ% Vessels (S2/S1)",
        outfile="BARCHART_PctChange_Vessels_by_Set.pdf",
        output_folder=output_folder,
    )

    # Plaque
    df_p = compute_pct_df(all_rows, "plaque_vol_um3")
    df_p.to_csv(os.path.join(output_folder, "PctChange_Plaques_by_Set.csv"), index=False)
    bar_percent_fixed(
        df_p,
        title="Δ% Plaques (Session 2 vs Session 1)",
        ylabel="Δ% Plaques (S2/S1)",
        outfile="BARCHART_PctChange_Plaques_by_Set.pdf",
        output_folder=output_folder,
    )

# Plot everything and save to desktop
def main():
    output_root = "/Users/elinestas/Desktop/"
    out_folder = os.path.join(output_root, "ALL_AREAS")
    os.makedirs(out_folder, exist_ok=True)

    all_rows = load_all_rows()
    if not all_rows:
        print("[WARN] No data found — check paths.")
        return

    plot_percent_change_barcharts(all_rows, out_folder)

if __name__ == "__main__":
    main()