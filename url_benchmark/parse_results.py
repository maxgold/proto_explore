import glob
import pickle
import pandas as pd

dirs = glob.glob("/home/maxgold/workspace/greene/exp_local/exp_local/2023*/*")

res = {}
res2 = {}

for f in dirs:
    try:
        cfg = pickle.load(open(f"{f}/cfg.json", "rb"))
        if cfg["action_repeat"] == 2:
            if cfg["type"] == "finetune":
                env = cfg["task"]
                if env not in res.keys():
                    res[env] = {}
                    res2[env] = {}
                name = cfg["agent"]["name"]
                if snapshot not in res[env].keys():
                    res[env][snapshot] = {}
                    res2[env][snapshot] = {}
                snapshot = cfg["snapshot_ts"]
                if name not in res[env][snapshot].keys():
                    res[env][snapshot][name] = []
                    res2[env][snapshot][name] = []

                stats = pd.read_csv(f"{f}/eval.csv")
                res[env][snapshot][name].append(stats["episode_reward"].values)
                res2[env][snapshot][name].append(stats["episode_reward"].values.max())
    except:
        pass


res3 = {}
for k1, v1 in res2.items():
    res3[k1] = res3.get(k1, {})
    for k2, v2 in v1.items():
        res3[k1][k2] = res3[k1].get(k2, {})
        for k3, v3 in v2.items():
            v3 = [v for v in v3 if v != "episode_reward"]
            res3[k1][k2][k3] = (np.mean(v3), np.std(v3))

res1t = {}
for k1, v1 in res.items():
    res1t[k1] = res1t.get(k1, {})
    for k2, v2 in v1.items():
        res1t[k1][k2] = res1t[k1].get(k2, {})
        for k3, v3 in v2.items():
            v3 = [v for v in v3 if (v.shape[0]==20) or (v.shape[0]==19)]
            #v3 = [v for v in v3 if v != "episode_reward"]
            res1t[k1][k2][k3] = (np.mean(v3, 0), np.std(v3, 0))

import pandas as pd

results = []

res3.pop("point_mass_maze_reach_bottom_right")

for k1, v1 in res3.items():
    for k2, v2 in v1.items():
        for k3, v3 in v2.items():
            results.append(
                {
                    "task": k1,
                    "algo": k2,
                    "snapshot": int(k3),
                    "reward_mu": v3[0],
                    "reward_sig": v3[1],
                }
            )

results = pd.DataFrame(results)
results.sort_values(["task","algo","snapshot"])


import matplotlib.pyplot as plt
for env in res1t.keys():
    for snapshot in [200000, 500000, 1000000]:
        plot_name = f"{env}_{snapshot}_line"
        stats = res1t[env][snapshot]
        plt.figure()
        stats = sorted(stats.items())
        names = [v[0] for v in stats]
        values = [v[1] for v in stats]
        #names = list(stats.keys())
        #values = np.array(list(stats.values()))
        lines = [v[0] for v in values]
        errors = [v[1] for v in values]
        upper_bar = [v[0] + v[1] for v in values]
        lower_bar = [v[0] - v[1] for v in values]
        colors = ["lightcoral", "goldenrod", "cornflowerblue", "olivedrab"]
        for i, l in enumerate(lines):
            if not np.isnan(l.sum()):
                if names[i] != 'aps':
                    plt.plot(range(len(l)), l, color=colors[i], alpha=0.5, linewidth=4, label=names[i])
                    plt.fill_between(range(len(l)), lower_bar[i], upper_bar[i], color=colors[i], alpha=0.5)
        plt.legend()
        plt.title(plot_name)
        plt.ylim([-100, 1000])
        plt.savefig(f"/home/maxgold/workspace/icml/lines/{plot_name}")
        plt.close()


for env in res3.keys():
    for snapshot in [200000, 500000, 1000000]:
        plot_name = f"{env}_{snapshot}"
        stats = res3[env][snapshot]
        plt.figure()
        names = list(stats.keys())
        values = np.array(list(stats.values()))
        plt.bar(range(len(stats)), values[:,0], tick_label=names, yerr=values[:,1])
        plt.title(plot_name)
        plt.show()
        plt.savefig(f"/home/maxgold/workspace/icml/bars/{plot_name}")
        plt.close()



for env, v2 in res2.items():
    print(env)
    for k, v in sorted(v2.items()):
        print(k, round(v, 2))

import tqdm
import numpy as np

hmaps = glob.glob(
    "/home/maxgold/workspace/greene/exp_local/exp_local/2023*/*/heatmap.pkl"
)
hmaps2 = []
hmap_res = {}
for hmap in tqdm.tqdm(hmaps):
    traincsv = hmap.replace("heatmap.pkl", "train.csv")
    xy = np.array(pickle.load(open(hmap, "rb")))
    cfg = hmap.replace("heatmap.pkl", "cfg.json")
    cfg = pickle.load(open(cfg, "rb"))
    agent = cfg["agent"]["name"]
    domain = cfg["domain"]
    try:
        print(xy.shape)
        if (xy.shape[0] >= 1e6) and (cfg["type"] == "pretrain"):
            hmaps2.append(hmap)
            if domain not in hmap_res.keys():
                hmap_res[domain] = {}
            if agent not in hmap_res[domain].keys():
                hmap_res[domain][agent] = []
            hmap_res[domain][agent].append((np.array(xy)[:, :2], cfg))
    except Exception as e:
        print(e)
        continue

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def make_heatmap(xy, path):
    heatmap, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=50, range=np.array(([-0.29, 0.29], [-0.29, 0.29]))
    )
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
    plt.savefig(path)


import os
import json

base_folder = "/home/maxgold/workspace/icml/heatmaps"

for domain, v1 in hmap_res.items():
    for agent, arrs in v1.items():
        for i, (arr, cfg) in enumerate(arrs):
            folder = f"{base_folder}/{domain}/{agent}/{i}"
            os.makedirs(folder, exist_ok=True)
            with open(f"{folder}/cfg.json", "w") as f:
                cfg["agent"] = dict(cfg["agent"])
                cfg["agent"]["obs_shape"] = list(cfg["agent"]["obs_shape"])
                cfg["agent"]["action_shape"] = list(cfg["agent"]["action_shape"])
                cfg["snapshots"] = list(cfg["snapshots"])
                json.dump(dict(cfg), f)
            for cut in (int(2e5), int(5e5), int(1e6), int(2e6)):
                make_heatmap(arr[:cut], f"{folder}/{cut}_hmap.png")


for hmap in hmaps2:
    print(np.array(xy).shape)
