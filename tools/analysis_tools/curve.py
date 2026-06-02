import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

log_path = Path("/home/lzq/mmdetection-3.3.0/work_dirs/dino-4scale_r50_8xb2-12e_bonai_bbox/20260529_182902/vis_data/scalars.json")

records = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.DataFrame(records)

print("Available keys:")
print(df.columns.tolist())

x_col = "step"

# 画 mAP 曲线
map_keys = [
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
]

plt.figure(figsize=(9, 5))

for key in map_keys:
    if key in df.columns:
        tmp = df.dropna(subset=[key])
        if not tmp.empty:
            plt.plot(tmp[x_col], tmp[key], marker="o", label=key)

plt.xlabel("Step")
plt.ylabel("Metric")
plt.title("Validation Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("map_curve.png", dpi=300)
plt.show()

# 画 loss 曲线
loss_keys = [
    "loss",

]

plt.figure(figsize=(9, 5))

for key in loss_keys:
    if key in df.columns:
        tmp = df.dropna(subset=[key])
        if not tmp.empty:
            plt.plot(tmp[x_col], tmp[key], label=key)

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()