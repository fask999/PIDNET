"""Check the runnable presets and real CLI entry points on synthetic data."""

import argparse
import gc
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

from configs import config, update_config
from models.pidnet import get_seg_model
from utils.ablation_config import (
    get_boundary_kwargs, get_boundary_loss_kwargs, get_model_kwargs,
    register_ablation_config,
)

SEED = 304
HEIGHT, WIDTH = 128, 192
PRESETS = {
    # Context, I atrous mode/rate, boundary target/mask/Dice.
    "baseline": ("auto", "none", 2, "legacy", False, 0.0),
    "pappm": ("pappm", "none", 2, "legacy", False, 0.0),
    "dappm": ("dappm", "none", 2, "legacy", False, 0.0),
    "pmpm": ("pmpm", "none", 2, "legacy", False, 0.0),
    "denseaspp": ("denseaspp", "none", 2, "legacy", False, 0.0),
    "valid-edge": ("auto", "none", 2, "valid", False, 0.0),
    "void-mask": ("auto", "none", 2, "legacy", True, 0.0),
    "dice": ("auto", "none", 2, "legacy", False, 1.0),
    "boundary": ("auto", "none", 2, "valid", True, 1.0),
    "fixed1": ("auto", "fixed", 1, "legacy", False, 0.0),
    "fixed2": ("auto", "fixed", 2, "legacy", False, 0.0),
    "fadc": ("auto", "fadc", 2, "legacy", False, 0.0),
    "combo-fixed": ("pmpm", "fixed", 2, "valid", True, 1.0),
    "combo-fadc": ("denseaspp", "fadc", 2, "valid", True, 1.0),
}


def cli_args(dataset, preset):
    env = dict(os.environ, PIDNET_PYTHON=sys.executable)
    env.pop("PIDNET_RUN_TAG", None)
    cmd = ["bash", str(EXP / "scripts/run_ablation.sh"), "show", dataset, preset]
    fields = shlex.split(subprocess.check_output(cmd, cwd=ROOT, env=env, text=True))
    cfg = fields[fields.index("--cfg") + 1]
    opts = fields[fields.index("--seed") + 2:]
    assert len(opts) % 2 == 0
    return cfg, dict(zip(opts[::2], opts[1::2]))


def flatten(opts):
    return [value for pair in opts.items() for value in pair]


def load_cfg(cfg, opts):
    local = register_ablation_config(config.clone())
    update_config(local, argparse.Namespace(cfg=cfg, opts=flatten(opts)))
    return local


def verify_presets():
    for dataset, classes in (("cityscapes", 19), ("camvid", 11)):
        for preset, expected in PRESETS.items():
            cfg, opts = cli_args(dataset, preset)
            local = load_cfg(cfg, opts)
            model = get_model_kwargs(local)
            boundary = get_boundary_kwargs(local)
            loss = get_boundary_loss_kwargs(local)
            observed = (
                model["context"], model["atrous_mode"], model["atrous_rate"],
                boundary["bd_target"], boundary["bd_mask_void"], loss["dice_weight"],
            )
            assert observed == expected, (dataset, preset, observed)
            assert boundary["bd_dilate_size"] == (3 if expected[3] == "valid" else 4)
            assert local.DATASET.NUM_CLASSES == classes
            assert local.DATASET.TRAIN_SET == "list/{}/train.lst".format(dataset)
            assert local.DATASET.TEST_SET == "list/{}/val.lst".format(dataset)
            assert (ROOT / local.OUTPUT_DIR).resolve() == EXP / "output" / dataset / preset
            assert (ROOT / local.LOG_DIR).resolve() == EXP / "logs" / dataset / preset
        print("{}: {} command presets select the expected independent options, train/val lists and output paths".format(
            dataset, len(PRESETS)), flush=True)


def synthetic_dataset(root, dataset):
    lists = root / "list" / dataset
    images = root / dataset / "images"
    labels = root / dataset / "labels"
    for directory in (lists, images, labels):
        directory.mkdir(parents=True)
    grid_y, grid_x = np.indices((HEIGHT, WIDTH))
    splits = {"val": [2, 3]}
    if dataset == "cityscapes":
        splits["train"] = [0, 1]
    for split, indices in splits.items():
        rows = []
        for index in indices:
            image_name = "sample-{}.png".format(index)
            image = np.stack(((grid_x * 3 + index * 7) % 256, (grid_y * 5) % 256,
                              (grid_x + grid_y) % 256), axis=-1).astype(np.uint8)
            assert cv2.imwrite(str(images / image_name), image)
            label = np.where(grid_x < 85 + index, 0, np.where(grid_y < 78, 1, 2))
            # A void island tests invalid regions without consuming all valid interfaces.
            label[10:34, 41:54] = 3
            if dataset == "cityscapes":
                raw = np.array([7, 8, 11, 0], dtype=np.uint8)[label]
                assert cv2.imwrite(str(labels / image_name), raw)
            else:
                colors = np.array([(0, 128, 192), (128, 0, 0), (64, 0, 128), (1, 2, 3)],
                                  dtype=np.uint8)
                Image.fromarray(colors[label]).save(str(labels / image_name))
            rows.append("images/{0} labels/{0}".format(image_name))
        (lists / (split + ".lst")).write_text("\n".join(rows) + "\n")
    return images


def run_tool(name, args, timeout=90, expected_error=None):
    cmd = [sys.executable, "-B", "tools/{}.py".format(name)] + list(args)
    print("+ " + " ".join(shlex.quote(arg) for arg in cmd), flush=True)
    res = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, timeout=timeout)
    print(res.stdout, end="", flush=True)
    if expected_error is None:
        res.check_returncode()
    else:
        assert res.returncode != 0 and expected_error in res.stdout, res.stdout
    return res.stdout


def set_smoke_paths(opts, base):
    out, logs = base / "output", base / "logs"
    out.mkdir(parents=True)
    logs.mkdir(parents=True)
    opts.update({
        "DATASET.ROOT": str(base / "data") + os.sep,
        "OUTPUT_DIR": str(out),
        "LOG_DIR": str(logs),
        "GPUS": "(0,)",
        "WORKERS": "0",
        "CUDNN.BENCHMARK": "False",
        "MODEL.PRETRAINED": "",
        "MODEL.NAME": "pidnet_s",
        "TRAIN.IMAGE_SIZE": "[{}, {}]".format(WIDTH, HEIGHT),
        "TRAIN.MULTI_SCALE": "False",
        "TRAIN.FLIP": "False",
        "TRAIN.END_EPOCH": "1",
        "TEST.IMAGE_SIZE": "[{}, {}]".format(WIDTH, HEIGHT),
    })
    return out


def assert_evaluation(log):
    match = re.search(r"MeanIU:\s*([\d.]+),\s*Pixel_Acc:\s*([\d.]+),\s*Mean_Acc:\s*([\d.]+)", log)
    assert match and all(math.isfinite(float(item)) for item in match.groups()), log


def verify_cli():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This CLI smoke requires one visible CUDA device; use --cpu-only for presets.")
    print("Synthetic CLI smoke (not dataset metrics): {}; torch {}, CUDA {}".format(
        torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda), flush=True)
    with tempfile.TemporaryDirectory(prefix="cli-smoke-", dir=str(EXP / "output")) as temp:
        temp = Path(temp)
        cfg, opts = cli_args("cityscapes", "combo-fadc")
        city = temp / "city"
        out = set_smoke_paths(opts, city)
        images = synthetic_dataset(city / "data", "cityscapes")
        log = run_tool("train", ["--cfg", cfg, "--seed", str(SEED)] + flatten(opts))
        assert "Epoch: [0/1] Iter:[0/1]" in log and "Iter:[1/1]" not in log
        boundary = re.search(r"BCE loss:\s*([\d.]+)", log)
        assert boundary and 0 < float(boundary[1]) < float("inf")
        path = out / "cityscapes" / Path(cfg).stem / "checkpoint.pth.tar"
        trained = torch.load(str(path), map_location="cpu")
        assert trained["epoch"] == 1 and trained["optimizer"]["state"]
        state = trained["state_dict"]
        for key in ("model.layer5.1.conv2.conv_offset.weight", "model.seghead_d.conv2.weight",
                    "model.final_layer.conv2.weight"):
            assert torch.isfinite(state[key]).all(), key
        assert any(k.startswith("model.spp.layers.") for k in state)
        assert "model._i_atrous_signature_fadc_s5_b1_r2" in state
        assert state["model.final_layer.conv2.weight"].shape[0] == 19
        del state, trained
        gc.collect()
        opts["TEST.MODEL_FILE"] = str(path)
        assert_evaluation(run_tool("eval", ["--cfg", cfg] + flatten(opts)))
        pred = city / "predictions"
        run_tool("custom", ["--cfg", cfg, "--p", str(path), "--r", str(images),
                            "--output-dir", str(pred)] + flatten(opts))
        assert sorted(p.name for p in pred.glob("*.png")) == sorted(p.name for p in images.glob("*.png"))
        for item in pred.glob("*.png"):
            with Image.open(str(item)) as image:
                assert image.size == (WIDTH, HEIGHT)
        print("Cityscapes-shaped synthetic data: one training step, 19-class strict eval reload, {} predictions".format(
            len(list(pred.glob("*.png")))), flush=True)

        cfg, opts = cli_args("camvid", "combo-fixed")
        cam = temp / "camvid"
        set_smoke_paths(opts, cam)
        synthetic_dataset(cam / "data", "camvid")
        model = get_seg_model(load_cfg(cfg, opts), imgnet_pretrained=False, load_pretrained=False)
        assert model.final_layer.conv2.weight.shape[0] == 11
        path = cam / "model-only.pt"
        torch.save(model.state_dict(), str(path))
        del model
        opts["TEST.MODEL_FILE"] = str(path)
        assert_evaluation(run_tool("eval", ["--cfg", cfg] + flatten(opts)))
        print("CamVid-shaped synthetic data: 11-class PMPM/fixed model-only eval reload", flush=True)
        opts["ABLATION.I_ATROUS.RATE"] = "4"
        rejected = run_tool("eval", ["--cfg", cfg] + flatten(opts),
                            expected_error="Checkpoint does not match model")
        for rate in (2, 4):
            assert "_i_atrous_signature_fixed_s5_b1_r{}".format(rate) in rejected
        print("CamVid fixed rate mismatch: strict eval rejected rate-2 weights with rate 4", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="check presets without launching CUDA tools")
    args = parser.parse_args()
    os.chdir(ROOT)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    verify_presets()
    if not args.cpu_only:
        verify_cli()
