# Fetus‑YOLO (BTP_2) — Beginner‑friendly Runbook

This repo contains a clean, unified pipeline to train, evaluate, export, and run inference. You run one block for a quick sanity run, or one for a full “conference‑ready” run.

What happens:
- Train creates weights at `runs/detect/.../weights/best.pt`.
- Eval writes JSON/CSV metrics and confusion matrix into `results/...`.
- Export produces ONNX models in `exports/...`.
- Inference runs adaptive/global/tiling on images or webcam.

---

## 1) Ubuntu — ONE‑SHOT SANITY (quick 10‑epoch train → eval → export → batch infer)

Copy‑paste the whole block:
```
sudo apt-get update && sudo apt-get install -y git python3-venv python3-dev build-essential
cd ~ && git clone https://github.com/Srinivas-Raghav-VC/yolov11-multi-organ-fetus.git BTP_2
cd BTP_2
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch torchvision torchaudio ultralytics onnx onnxruntime opencv-python numpy PyYAML matplotlib tqdm pandas scikit-learn pycocotools

DATA=fetus-yolo/data_yolo/fetal_fp23.yaml
python scripts/train_cli.py --mode quick --data $DATA --img 896 --epochs 10 --device auto --yes
WEIGHTS=runs/detect/proposed_full/weights/best.pt
python scripts/eval_cli.py --weights $WEIGHTS --data $DATA --img 896 --conf 0.25 --device auto --save-dir results/eval_quick --coco
python scripts/export_cli.py --weights $WEIGHTS --img 896 --formats onnx --half --out exports
python scripts/infer_cli.py --onnx exports/onnx_fp16/weights/best.onnx --source fetus-yolo/data_yolo/images/val --mode adaptive --imgsz 896 --tile 640 --overlap 0.25 --conf 0.25 --device auto
```
Expected outputs: best.pt, evaluation_results.json/CSV, exported ONNX.

---

## 2) Ubuntu — ONE‑SHOT CONFERENCE (full 200‑epoch train → eval → export → batch infer)

Use the same venv as above (reactivate if in a new shell):
```
cd ~/BTP_2 && source .venv/bin/activate
DATA=fetus-yolo/data_yolo/fetal_fp23.yaml
python scripts/train_cli.py --mode proposed --data $DATA --img 1024 --epochs 200 --device auto --yes
WEIGHTS=runs/detect/proposed_full/weights/best.pt
python scripts/eval_cli.py --weights $WEIGHTS --data $DATA --img 1024 --conf 0.25 --device auto --save-dir results/eval_conference --coco
python scripts/export_cli.py --weights $WEIGHTS --img 1024 --formats onnx --half --out exports
python scripts/infer_cli.py --onnx exports/onnx_fp16/weights/best.onnx --source fetus-yolo/data_yolo/images/val --mode adaptive --imgsz 1024 --tile 640 --overlap 0.25 --conf 0.25 --device auto
```

---

## 3) Google Colab — run everything and save to Google Drive automatically

In Colab: Runtime → Change runtime type → GPU.

Mount Drive (saves checkpoints/exports permanently):
```
from google.colab import drive
drive.mount('/content/drive')
```

Clone + install:
```
%cd /content
!git clone https://github.com/Srinivas-Raghav-VC/yolov11-multi-organ-fetus.git BTP_2
%cd /content/BTP_2
!pip -q install ultralytics onnx onnxruntime opencv-python numpy PyYAML matplotlib tqdm pandas scikit-learn pycocotools
```

SANITY (saves to Drive):
```
DATA="fetus-yolo/data_yolo/fetal_fp23.yaml"
!python scripts/train_cli.py --mode quick --data $DATA --img 896 --epochs 10 --device 0 --yes --project \
  "/content/drive/MyDrive/BTP2_runs/detect"
WEIGHTS="/content/drive/MyDrive/BTP2_runs/detect/proposed_full/weights/best.pt"
!python scripts/eval_cli.py --weights $WEIGHTS --data $DATA --img 896 --conf 0.25 --device 0 --save-dir \
  "/content/drive/MyDrive/BTP2_results/eval_quick" --coco
!python scripts/export_cli.py --weights $WEIGHTS --img 896 --formats onnx --half --out \
  "/content/drive/MyDrive/BTP2_exports"
!python scripts/infer_cli.py --onnx \
  "/content/drive/MyDrive/BTP2_exports/onnx_fp16/weights/best.onnx" \
  --source fetus-yolo/data_yolo/images/val --mode adaptive --imgsz 896 --tile 640 --overlap 0.25 --conf 0.25 --device 0
```

CONFERENCE (saves to Drive):
```
DATA="fetus-yolo/data_yolo/fetal_fp23.yaml"
!python scripts/train_cli.py --mode proposed --data $DATA --img 1024 --epochs 200 --device 0 --yes --project \
  "/content/drive/MyDrive/BTP2_runs/detect"
WEIGHTS="/content/drive/MyDrive/BTP2_runs/detect/proposed_full/weights/best.pt"
!python scripts/eval_cli.py --weights $WEIGHTS --data $DATA --img 1024 --conf 0.25 --device 0 --save-dir \
  "/content/drive/MyDrive/BTP2_results/eval_conference" --coco
!python scripts/export_cli.py --weights $WEIGHTS --img 1024 --formats onnx --half --out \
  "/content/drive/MyDrive/BTP2_exports"
!python scripts/infer_cli.py --onnx \
  "/content/drive/MyDrive/BTP2_exports/onnx_fp16/weights/best.onnx" \
  --source fetus-yolo/data_yolo/images/val --mode adaptive --imgsz 1024 --tile 640 --overlap 0.25 --conf 0.25 --device 0
```

Where outputs go on Drive:
- Checkpoints: `/content/drive/MyDrive/BTP2_runs/detect/.../weights/best.pt`
- Results: `/content/drive/MyDrive/BTP2_results/...`
- Exports: `/content/drive/MyDrive/BTP2_exports/...`

---

## 4) (Optional) Weights & Biases (W&B) — real‑time metrics & checkpoint upload

Enable W&B once, then train as usual.

Ubuntu:
```
cd ~/BTP_2 && source .venv/bin/activate
python -m pip install wandb
export WANDB_API_KEY=YOUR_KEY
export WANDB_PROJECT=fetus-yolo
python - << 'PY'
from ultralytics.utils import SETTINGS
SETTINGS.update({'wandb': True})
print('W&B enabled in Ultralytics SETTINGS')
PY
```

Colab:
```
!pip -q install wandb
!wandb login YOUR_KEY
python - << 'PY'
from ultralytics.utils import SETTINGS
SETTINGS.update({'wandb': True})
print('W&B enabled in Ultralytics SETTINGS')
PY
```

Upload checkpoints folder as a W&B artifact (after training):
```
python - << 'PY'
import os, wandb
run = wandb.init(project=os.environ.get('WANDB_PROJECT','fetus-yolo'), job_type='checkpoints', anonymous='allow')
art = wandb.Artifact(name='fetus-yolo-checkpoints', type='checkpoints')
RUN_DIR = 'runs/detect/proposed_full/weights'  # change if your run name differs
art.add_dir(RUN_DIR)
run.log_artifact(art)
run.finish()
print('Uploaded checkpoints from', RUN_DIR)
PY
```

---

## 5) Push this repo to your GitHub (once)

```
sudo apt-get update && sudo apt-get install -y git
cd ~/BTP_2
git init
git remote add origin https://github.com/Srinivas-Raghav-VC/yolov11-multi-organ-fetus.git
git add -A
git commit -m "Initial import: cleaned unified Fetus-YOLO (BTP_2)"
git branch -M main
git push -u origin main
```
