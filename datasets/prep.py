import os
import glob

base = [f.split("/")[-1] for f in glob.glob("/Users/dhruv/Downloads/faces/*.jpg")]
base.sort()

os.makedirs("/Users/dhruv/Documents/faces2comics/train/faces")
os.makedirs("/Users/dhruv/Documents/faces2comics/train/comics")
os.makedirs("/Users/dhruv/Documents/faces2comics/val/faces")
os.makedirs("/Users/dhruv/Documents/faces2comics/val/comics")

num_train = int(0.9 * len(base))
train_f, val_f = base[:num_train], base[num_train:]

for f in train_f:
    src = f"/Users/dhruv/Downloads/faces/{f}"
    tgt = f"/Users/dhruv/Documents/faces2comics/train/faces/{f}"
    _ = os.system(f"cp {src} {tgt}")
    src = f"/Users/dhruv/Downloads/comics/{f}"
    tgt = f"/Users/dhruv/Documents/faces2comics/train/comics/{f}"
    _ = os.system(f"cp {src} {tgt}")
for f in val_f:
    src = f"/Users/dhruv/Downloads/faces/{f}"
    tgt = f"/Users/dhruv/Documents/faces2comics/val/faces/{f}"
    _ = os.system(f"cp {src} {tgt}")
    src = f"/Users/dhruv/Downloads/comics/{f}"
    tgt = f"/Users/dhruv/Documents/faces2comics/val/comics/{f}"
    _ = os.system(f"cp {src} {tgt}")