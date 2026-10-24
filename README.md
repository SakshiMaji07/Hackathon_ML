# Hackathon_ML local pipeline test

## Current project status

The repository contains capture, alpha inference, and compositing modules, and this checkout includes two raw video clips under `data/raw`. It does not include a trained checkpoint, paired foreground/alpha training data, or a background image. The `models/weights`, `data/raw`, and `data/processed` directories are intentionally ignored by Git. Supply a compatible `.pth` checkpoint and background image for the desktop photobooth; the standalone video visualizer can use a solid background.

The current model builder is a small alpha-prediction network. It is not the upstream Robust Video Matting model unless its checkpoint was trained for this exact architecture. The alpha-only output now composites the original camera pixels as foreground. CPU defaults use a reduced model scale, fixed subject order, and no MiDaS model download.

## CPU-only setup

Use Python 3.10 or 3.11 in a virtual environment. Install PyTorch and TorchVision using the CPU wheel instructions for your operating system from [PyTorch's install selector](https://pytorch.org/get-started/locally/), then install the remaining packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For first-time UI validation, run synthetic mode (no cameras, weights, or background required):

```powershell
python local_test_backend.py --synthetic
```

Open http://127.0.0.1:8000. This checks browser streaming and live stats only; it does not run the CV model.

For real model inference with two webcams:

```powershell
python local_test_backend.py --input-a 0 --input-b 1 --model models/weights/best_model.pth --background data/raw/background.jpg --width 640 --height 360 --scale 0.5 --target-fps 8 --threads 4
```

For recorded clips, replace `0` and `1` with quoted video paths. Two local clips are the most repeatable way to tune performance before testing physical webcams. The server loops a file at end-of-stream. Open http://127.0.0.1:8000 to view both inputs and the composite. `Replaced frames` counts camera frames overwritten while inference was busy; this is expected with latest-frame processing and indicates load, not a dropped HTTP connection.

The backend binds to localhost by default. It serves the HTML interface at `/`, three MJPEG feeds at `/feed/a`, `/feed/b`, and `/feed/output`, and metrics at `/stats`.

## Desktop app

After placing a compatible checkpoint and background at the paths in `config/config.yaml`, run:

```powershell
python main.py --config config/config.yaml
```

Press `q` in the OpenCV window to exit. The default config uses CPU, 640×360, a 0.5 model scale, and fixed A-over-B order. Turn on `depth.enabled` only when you specifically want to evaluate MiDaS ordering; it downloads and runs a separate model and is likely too slow for this CPU setup.

## Checks and limitations

Run the pure unit tests with:

```powershell
python -m pytest tests/test_matting.py tests/test_composite.py
```

`tests/test_capture.py` needs two working cameras and can be run separately. Its default 30 FPS assertion measures the reader loop, not full-model processing throughput. This checkout has raw clips but no labelled train/validation dataset or trained checkpoint, so model evaluation and real checkpoint inference require those assets first. The synthetic mode does not claim to validate alpha quality or real model performance.

For a website backend, use this Python service as a local integration harness first. Browser camera capture, WebGPU inference, and the production service protocol are separate integration work: this harness reads host webcams/files with OpenCV, runs PyTorch on the CPU, and sends MJPEG over HTTP. A WebGPU hosting environment does not automatically accelerate this Python/PyTorch process. Profile with two recorded clips at 640×360 and 0.5 scale, then reduce scale/FPS further if measured throughput is below the target. Keep capture and inference decoupled with latest-frame buffers as implemented here.

## Train, evaluate, and run the matting model

The video files used by the photobooth are inference inputs; they do not contain the ground-truth alpha masks required to train this supervised model. Training data must contain paired foreground images and alpha mattes. The loader matches foreground/matte files by filename stem (the extensions may differ), and backgrounds are sampled independently for synthetic compositing.

Create this directory structure (images can be PNG, JPEG, TIFF, BMP, or WebP):

```text
data/matting/
  train/
    foregrounds/  # RGB source images, e.g. person_0001.jpg
    alphas/       # matching grayscale/alpha mattes, e.g. person_0001.png
    backgrounds/  # unrelated background images
  val/
    foregrounds/
    alphas/
    backgrounds/
```

Keep validation subjects separate from training subjects to measure generalization. Foregrounds and mattes in each split must have matching stems and image dimensions. Each batch must have identical spatial dimensions; the supplied training configuration uses batch size 1 so different source image sizes can be loaded across batches. Use high-quality soft alpha mattes where available, rather than masks generated from the input video itself.

The default `config/training.yaml` starts from an ImageNet-pretrained MobileNetV3-small feature extractor and learns a new alpha head. The first run downloads the torchvision weights if they are not already cached. If the machine is offline, set `model.pretrained_backbone: false`; training will then start all weights from random initialization and generally needs more data and training time. This repository's model is a lightweight alpha predictor, not the upstream Robust Video Matting architecture.

1. Install dependencies in the active virtual environment. Install the PyTorch/TorchVision pair using the official PyTorch selector for your CPU or CUDA runtime, then install the project requirements:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

2. Edit `config/training.yaml` if your data is elsewhere. Set `device` to `cuda` only when a compatible CUDA-enabled PyTorch build and NVIDIA GPU are available. `cpu` is supported but training will be slower.

3. For a first smoke run, temporarily set `training.epochs: 1` in the YAML to confirm the paths, checkpoint, and batches are correct. Then restore your desired epoch count and train:

   ```powershell
   python -m training.train_matting --config config/training.yaml
   ```

   Checkpoints are written to `models/weights/latest_checkpoint.pth` and `models/weights/best_model.pth`. The trainer's validation split uses no random affine, color-jitter, or flip augmentation.

4. Evaluate the best checkpoint using the held-out validation split:

   ```powershell
   python -m training.evaluate_matting `
     --config config/training.yaml `
     --checkpoint models/weights/best_model.pth `
     --split val `
     --output-dir models/weights/val_alpha
   ```

   The command prints combined loss, alpha MAE, and alpha MSE; the optional output directory receives grayscale predicted alpha PNGs. Inspect these predictions visually as well as checking metrics. `--split` can select another split configured under `data`, such as `test`.

5. Test inference on two clips using the saved checkpoint:

   ```powershell
   python visualize_matting.py `
     --video-a data/raw/video_a.mp4 `
     --video-b data/raw/video_b.mp4 `
     --model models/weights/best_model.pth `
     --backbone mobilenet_v3_small `
     --device cpu `
     --width 640 --height 360 --scale 0.5
   ```

   Press `q` or `Esc` in an OpenCV window to exit. Use the same backbone as `model.backbone` in the training config. The visualizer uses a solid background and fixed A-over-B ordering; it is for checking inference/mattes, not evaluating depth estimation or photobooth enhancements.

To continue training from the latest checkpoint, set `training.resume_from_checkpoint: models/weights/latest_checkpoint.pth` in the YAML config. For TensorBoard logs, run `tensorboard --logdir runs/matting` and open the URL it prints.
