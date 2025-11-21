# -*- coding: utf-8 -*-
"""
Final version:
Left = previous Sample A/B percent-change plot
Right = 3 mice absolute consecutive change plot (with Mean ± SEM)
Only right-plot lines have transparency alpha=0.7
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# Part 1: Sample A/B (unchanged)
# =========================
sample_a = np.array([
    [3.5964, 4.2768, 4.55868],
    [3.8880, 4.19904, 4.19904],
    [5.5404, 5.86116, 6.83316],
    [5.86116, 6.18192, 6.18192],
    [8.78688, 9.85608, 10.7406]
])
x_a = np.array([32, 37, 42])

sample_b = np.array([
    [11.72232, 11.72232, 10.7406],
    [9.7686,   10.7406,  11.72232],
    [8.78688,  9.7686,   7.81488],
    [8.7966,   9.7686,   8.78688],
    [7.80516,  7.81488,  8.78688]
])
x_b = np.array([31, 40, 45])

def stats(data):
    mean = np.nanmean(data, axis=0)
    sd   = np.nanstd(data, axis=0, ddof=1)
    sem  = sd / np.sqrt(np.sum(~np.isnan(data), axis=0))
    return mean, sd, sem

# relative-to-first-point percent change
sample_a_pct = (sample_a - sample_a[:, [0]]) / sample_a[:, [0]] * 100
sample_b_pct = (sample_b - sample_b[:, [0]]) / sample_b[:, [0]] * 100

mean_a_pct, sd_a_pct, sem_a_pct = stats(sample_a_pct)
mean_b_pct, sd_b_pct, sem_b_pct = stats(sample_b_pct)

# X 轴归一化（同你旧代码）
x_norm_a = ((x_a / 100.0) / 0.45) * 100
x_norm_b = ((x_b / 100.0) / 0.45) * 100


# =========================
# Part 2: 3 mice Depth–Shift data
# =========================
mouse1 = np.array([
    [80, 0],[85, 0],[90, -0.4885],[95, 0.4885],[100, 0.4885],[105, 0],[110, 0.4885],
    [185, 0],[190, -0.977],[195, 0],[200, 0],[205, 0],[210, 0.977],
    [260, 0],[265, 0],[270, 0.977],[275, -0.977],[280, 0.4885],[285, 0],[290, 0.977],
    [550, 0],[555, 0.977],[560, 0.977],[565, -0.4885],[570, 0],[575, 0.4885],
    [785, 0],[790, -0.977],[795, 0],[800, 1.4655],[805, -1.4655],[810, 0.4885],[815, 0.977]
])

mouse2 = np.array([
    [40, 0],[45, 0.977],[50, -0.977],[55, -0.4885],[60, 0],[65, 0],[70, 0],[75, 0.977],[80, -0.4885],
    [235, 0],[240, 0.4885],[245, 0.977],[250, -0.4885],[255, 0.4885],[260, -0.4885],
    [430, 0],[435, -0.4885],[440, 1.954],[445, 0],[450, -0.977],[455, 0.4885],[460, 0],[465, 0],
    [525, 0],[530, 0],[535, 0.977],[540, -0.4885],
    [285, 0],[290, 0.4885],[295, -0.4885],[300, 1.954],[305, -0.4885],[310, 0],[315, 0],[320, -0.4885],[325, 0.4885]
])

mouse3 = np.array([
    [35, 0],[40, 1.4655],[45, -0.4885],[50, -1.954],[55, 0],[60, -0.4885],[65, -0.4885],[70, 0],
    [200, 0],[205, 0.4885],[210, -0.4885],[215, 0.4885],[220, 0],[225, 0.4885],
    [470, 0],[475, 1.4655],[480, 0],[485, 0.4885],[490, 0.4885],[495, 0],[500, -0.4885],
    [580, 0],[585, 0],[590, -0.4885],[595, 0],[600, 0],[605, -0.4885],[610, 0],
    [830, 0],[835, 1.4655],[840, -0.4885],[845, -0.4885],[850, 0.977],[855, -0.4885]
])

mice = [mouse1, mouse2, mouse3]

# ---- absolute consecutive change ----
def consecutive_abs_change(y):
    y = np.asarray(y, float)
    d = np.zeros_like(y)
    d[1:] = y[1:] - y[:-1]
    return d

mice_depth_abs = []
for m in mice:
    depth = m[:, 0]
    shift = m[:, 1]
    abs_chg = consecutive_abs_change(shift)
    mice_depth_abs.append((depth, abs_chg))

# ---- Align depths for Mean ± SEM ----
all_depths = np.unique(np.concatenate([d for d, _ in mice_depth_abs]))
all_depths.sort()

abs_matrix = np.full((len(mice), len(all_depths)), np.nan)

for mi, (depth, abs_chg) in enumerate(mice_depth_abs):
    for di, d in enumerate(depth):
        idx = np.where(all_depths == d)[0][0]
        abs_matrix[mi, idx] = abs_chg[di]

mean_abs, sd_abs, sem_abs = stats(abs_matrix)

# =========================
# Part 3: Plot side by side
# =========================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- Left: Sample A/B ----
axL = axes[0]

la, = axL.plot(x_norm_a, mean_a_pct, '-o', label='Sample A', linewidth=2)
axL.fill_between(x_norm_a, mean_a_pct - sem_a_pct, mean_a_pct + sem_a_pct,
                 alpha=0.25, color=la.get_color())

lb, = axL.plot(x_norm_b, mean_b_pct, '-o', label='Sample B', linewidth=2)
axL.fill_between(x_norm_b, mean_b_pct - sem_b_pct, mean_b_pct + sem_b_pct,
                 alpha=0.25, color=lb.get_color())

axL.set_xlabel("Normalized Power (%)")
axL.set_ylabel("Change from first point (%)")
axL.set_ylim([-30, 30])
axL.legend(loc='lower right', frameon=False)

axL.spines['top'].set_visible(False)
axL.spines['right'].set_visible(False)
axL.spines['left'].set_visible(True)
axL.spines['bottom'].set_visible(True)

# ---- Right: absolute change (with alpha=0.7) ----
axR = axes[1]

# individual mice (alpha=0.7)
for i, (depth, abs_chg) in enumerate(mice_depth_abs, start=1):
    axR.plot(depth, abs_chg, '-o',
             linewidth=1.5, markersize=3,
             alpha=0.7,                        # ★ 唯一要求
             label=f"Mouse {i}")

# mean line (alpha=0.7)
lm, = axR.plot(all_depths, mean_abs, '-o',
               linewidth=2.2, markersize=3.5,
               alpha=0.7,                      # ★ 唯一要求
               label="Mean")

# SEM shading stays alpha=0.25
axR.fill_between(all_depths,
                 mean_abs - sem_abs,
                 mean_abs + sem_abs,
                 alpha=0.25,
                 color=lm.get_color(),
                 label="±SEM")

axR.set_xlabel("Depth (µm)")
axR.set_ylabel("Shift (µm)")
axR.legend(loc='lower right', frameon=False)

axR.spines['top'].set_visible(False)
axR.spines['right'].set_visible(False)
axR.spines['left'].set_visible(True)
axR.spines['bottom'].set_visible(True)

fig.tight_layout()

# =========================
# Save
# =========================
save_dir = r"C:\Users\myang\Desktop\3p slice roi"
os.makedirs(save_dir, exist_ok=True)

fig.savefig(os.path.join(save_dir,
    "combined.pdf"),
    format="pdf", dpi=600)

plt.show()
