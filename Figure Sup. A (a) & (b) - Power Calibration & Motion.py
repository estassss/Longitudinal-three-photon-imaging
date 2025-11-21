import numpy as np
import matplotlib.pyplot as plt
import os



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
    n_valid = np.sum(~np.isnan(data), axis=0)
    sem  = sd / np.sqrt(n_valid)
    sem[n_valid <= 1] = 0.0
    return mean, sd, sem


sample_a_pct = (sample_a - sample_a[:, [0]]) / sample_a[:, [0]] * 100
sample_b_pct = (sample_b - sample_b[:, [0]]) / sample_b[:, [0]] * 100

mean_a_pct, sd_a_pct, sem_a_pct = stats(sample_a_pct)
mean_b_pct, sd_b_pct, sem_b_pct = stats(sample_b_pct)

x_norm_a = ((x_a / 100.0) / 0.45) * 100
x_norm_b = ((x_b / 100.0) / 0.45) * 100




mouse1 = np.array([
    [80,0],[85,0],[90,-0.4885],[95,0.4885],[100,0.4885],[105,0],[110,0.4885],
    [185,0],[190,-0.977],[195,0],[200,0],[205,0],[210,0.977],
    [260,0],[265,0],[270,0.977],[275,-0.977],[280,0.4885],[285,0],[290,0.977],
    [550,0],[555,0.977],[560,0.977],[565,-0.4885],[570,0],[575,0.4885],
    [785,0],[790,-0.977],[795,0],[800,1.4655],[805,-1.4655],[810,0.4885],[815,0.977]
])

mouse2 = np.array([
    [40,0],[45,0.977],[50,-0.977],[55,-0.4885],[60,0],[65,0],[70,0],[75,0.977],[80,-0.4885],
    [235,0],[240,0.4885],[245,0.977],[250,-0.4885],[255,0.4885],[260,-0.4885],
    [430,0],[435,-0.4885],[440,1.954],[445,0],[450,-0.977],[455,0.4885],[460,0],[465,0],
    [525,0],[530,0],[535,0.977],[540,-0.4885],
    [285,0],[290,0.4885],[295,-0.4885],[300,1.954],[305,-0.4885],[310,0],[315,0],[320,-0.4885],[325,0.4885]
])

mouse3 = np.array([
    [35,0],[40,1.4655],[45,-0.4885],[50,-1.954],[55,0],[60,-0.4885],[65,-0.4885],[70,0],
    [200,0],[205,0.4885],[210,-0.4885],[215,0.4885],[220,0],[225,0.4885],
    [470,0],[475,1.4655],[480,0],[485,0.4885],[490,0.4885],[495,0],[500,-0.4885],
    [580,0],[585,0],[590,-0.4885],[595,0],[600,0],[605,-0.4885],[610,0],
    [830,0],[835,1.4655],[840,-0.4885],[845,-0.4885],[850,0.977],[855,-0.4885]
])

mice = [mouse1, mouse2, mouse3]

def split_by_gap_adaptive(depth, values, factor=2.0):
    depth = np.asarray(depth)
    values = np.asarray(values)

    order = np.argsort(depth)
    depth = depth[order]
    values = values[order]

    diffs = np.diff(depth)
    if len(diffs) == 0:
        return [(depth, values)]

    small = diffs[diffs <= np.median(diffs)*3]
    typical = np.median(small) if len(small)>0 else np.median(diffs)
    gap_thr = typical * factor

    cut = np.where(diffs > gap_thr)[0]

    segs = []
    start = 0
    for ci in cut:
        end = ci + 1
        segs.append((depth[start:end], values[start:end]))
        start = end

    segs.append((depth[start:], values[start:]))
    return segs




fig, axes = plt.subplots(1, 2, figsize=(14, 5))


axL = axes[0]

la, = axL.plot(x_norm_a, mean_a_pct, '-o', linewidth=2, label='Sample A')
axL.fill_between(x_norm_a, mean_a_pct - sem_a_pct, mean_a_pct + sem_a_pct,
                 alpha=0.25, color=la.get_color())

lb, = axL.plot(x_norm_b, mean_b_pct, '-o', linewidth=2, label='Sample B')
axL.fill_between(x_norm_b, mean_b_pct - sem_b_pct, mean_b_pct + sem_b_pct,
                 alpha=0.25, color=lb.get_color())

axL.spines['top'].set_visible(False)
axL.spines['right'].set_visible(False)
axL.set_xlabel("Normalized Power (%)")
axL.set_ylabel("Change from first point (%)")
axL.set_ylim([-40,40])
axL.legend(loc='lower right', frameon=False)



axR = axes[1]

mouse_colors = ["tab:blue", "tab:orange", "tab:green"]

for mi, m in enumerate(mice):
    color = mouse_colors[mi]
    depth = m[:,0]
    shift = m[:,1]
    segs = split_by_gap_adaptive(depth, shift, factor=2.0)

    first_label = True
    for dseg, yseg in segs:
        axR.plot(
            dseg, yseg, '-o',
            color=color,
            linewidth=1.5,
            markersize=3,
            alpha=0.7,
            label=f"Mouse {mi+1}" if first_label else None
        )
        first_label = False

axR.spines['top'].set_visible(False)
axR.spines['right'].set_visible(False)
axR.set_xlabel("Depth (µm)")
axR.set_ylabel("Shift (µm)")
axR.legend(loc='lower right', frameon=False)

fig.tight_layout()

# 保存
save_dir = r"C:\Users\myang\Desktop\3p slice roi"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(os.path.join(save_dir,"cimbined-1.pdf"), dpi=600)

plt.show()
