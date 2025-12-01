# handler.py  -- RunPod serverless worker for training

import time
from typing import Any, Dict, List

import runpod  # pip install runpod

# Optional: try PyTorch for real GPU training
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def _get_gpu_meta() -> Dict[str, Any]:
    """
    Called inside the RunPod container.
    We'll just ask PyTorch what device it thinks it's on.
    """
    meta: Dict[str, Any] = {}

    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        meta["name"] = props.name
        meta["memory_gb"] = round(props.total_memory / (1024**3), 1)
    else:
        meta["name"] = "CPU or unknown"
        meta["memory_gb"] = None

    return meta


def _train_demo(heavy: bool, logs: List[str]) -> Dict[str, Any]:
    """
    Demo training loop:

    - If PyTorch is available, train a tiny model on random data.
    - Otherwise, run a dummy loop that still gives meaningful logs.

    You can later replace this with your real "trend engine" training
    routine that uses OHLC data, etc.
    """
    start = time.time()

    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device("cuda")
        logs.append(f"Using device: {device}")

        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # e.g., buy/hold/sell logits
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # fake dataset: 2048 examples of 32-dim features
        x = torch.randn(2048, 32, device=device)
        y = torch.randint(0, 3, (2048,), device=device)

        epochs = 3 if not heavy else 10
        logs.append(f"Training for {epochs} epochs (demo).")

        loss_start = None
        loss_end = None

        for epoch in range(1, epochs + 1):
            model.train()
            opt.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            loss_val = float(loss.item())
            if loss_start is None:
                loss_start = loss_val
            loss_end = loss_val

            if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
                logs.append(
                    f"[epoch {epoch}/{epochs}] loss={loss_val:.4f}"
                )

        duration = time.time() - start

        # TODO: save weights to some persistent storage (S3, etc.) if you want
        return {
            "epochs": epochs,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "duration_sec": duration,
            "heavy": heavy,
        }

    # ----------------- dummy CPU only -----------------
    logs.append("PyTorch/CUDA not available, running dummy CPU loop.")
    steps = 40 if not heavy else 100
    loss = 1.0
    loss_start = loss

    for step in range(1, steps + 1):
        time.sleep(0.1)
        loss *= 0.985

        if step == 1 or step % 8 == 0 or step == steps:
            logs.append(f"[dummy step {step}/{steps}] loss≈{loss:.4f}")

    duration = time.time() - start
    return {
        "epochs": steps,
        "loss_start": loss_start,
        "loss_end": loss,
        "duration_sec": duration,
        "heavy": heavy,
        "mode": "dummy",
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod entrypoint.

    The backend calls:
        POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
        {
          "input": {
            "heavy": true/false,
            ...
          }
        }

    Here we read event["input"], run training, and return logs + summary.
    """
    logs: List[str] = []
    logs.append("🔥 RunPod training worker started.")

    job_input = event.get("input") or {}
    heavy = bool(job_input.get("heavy", False))

    logs.append(f"heavy={heavy}")

    gpu = _get_gpu_meta()
    logs.append(f"GPU: {gpu.get('name')} ({gpu.get('memory_gb')} GB)")

    summary = _train_demo(heavy, logs)

    headline = (
        f"Training done on {gpu.get('name')} | "
        f"loss {summary.get('loss_start'):.4f} → {summary.get('loss_end'):.4f}"
        if summary.get("loss_start") is not None and summary.get("loss_end") is not None
        else "Training run finished."
    )

    summary["headline"] = headline

    # IMPORTANT: We don't compute cost here; RunPod's API will add a "cost"
    # number in the outer response. The backend reads that and translates
    # it into dollars + GPU hours.

    return {
        "logs": logs,
        "summary": summary,
        "gpu": gpu,
    }


runpod.serverless.start({"handler": handler})
