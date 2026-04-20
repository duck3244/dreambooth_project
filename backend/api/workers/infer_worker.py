"""Subprocess entry point for inference.

Usage:
    python -m backend.api.workers.infer_worker \\
        --request /path/to/request.json \\
        --event-log /path/to/events.jsonl \\
        --output-dir /path/to/output

The request JSON contains: base_model, model_path, model_kind ("lora" or
"full"), prompts, negative_prompt, num_inference_steps, guidance_scale,
height, width, num_images_per_prompt, seed.

Events: started / image / completed / error — written as JSONL for the
API layer to tail and stream as SSE.
"""

import sys

# xformers guard — identical rationale to train_worker.py.
import os as _os
if _os.environ.get("DREAMBOOTH_ENABLE_XFORMERS", "0") != "1":
    sys.modules["xformers"] = None
    sys.modules["xformers.ops"] = None

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_BACKEND_DIR = _THIS.parents[2]
_PROJECT_ROOT = _BACKEND_DIR.parent
for p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _append_event(event_log: str, **fields: Any) -> None:
    try:
        Path(event_log).parent.mkdir(parents=True, exist_ok=True)
        with open(event_log, "a", encoding="utf-8") as f:
            rec = {"ts": time.time(), **fields}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    except Exception:
        pass


def _safe_slug(text: str, limit: int = 40) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch == " ":
            keep.append("_")
    s = "".join(keep).strip("_-") or "img"
    return s[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description="DreamBooth inference subprocess worker")
    ap.add_argument("--request", required=True, help="Path to request JSON")
    ap.add_argument("--event-log", dest="event_log", required=True, help="Path to JSONL event log")
    ap.add_argument("--output-dir", dest="output_dir", required=True, help="Directory for generated images")
    args = ap.parse_args()

    try:
        with open(args.request, "r", encoding="utf-8") as f:
            req = json.load(f)
    except Exception as e:
        _append_event(args.event_log, type="error", error=str(e), error_type=type(e).__name__)
        traceback.print_exc()
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model: str = req["base_model"]
    model_path: str = req["model_path"]
    model_kind: str = req["model_kind"]
    prompts: list = req["prompts"]
    negative_prompt = req.get("negative_prompt") or None
    num_inference_steps = int(req.get("num_inference_steps", 30))
    guidance_scale = float(req.get("guidance_scale", 7.5))
    height = int(req.get("height", 512))
    width = int(req.get("width", 512))
    num_images_per_prompt = int(req.get("num_images_per_prompt", 1))
    seed = req.get("seed")

    total = len(prompts) * num_images_per_prompt
    start_t = time.time()

    _append_event(
        args.event_log,
        type="started",
        model_path=model_path,
        model_kind=model_kind,
        prompts=prompts,
        total=total,
    )

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except Exception as e:
        _append_event(args.event_log, type="error", error=str(e), error_type=type(e).__name__)
        traceback.print_exc()
        return 3

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if model_kind == "full":
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            # LoRA: load the base then attach the adapter.
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            # weight_name is required because we don't use safetensors.
            pipe.load_lora_weights(model_path, weight_name="pytorch_lora_weights.bin")

        pipe = pipe.to(device)
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    except Exception as e:
        _append_event(args.event_log, type="error", error=str(e), error_type=type(e).__name__)
        traceback.print_exc()
        return 4

    index = 0
    try:
        for pi, prompt in enumerate(prompts):
            for k in range(num_images_per_prompt):
                this_seed = int(seed) + index if seed is not None else int(time.time() * 1000) & 0xFFFFFFFF
                generator = torch.Generator(device=device).manual_seed(this_seed)

                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                )
                image = result.images[0]

                filename = f"{index:03d}_{_safe_slug(prompt)}_{this_seed}.png"
                image.save(output_dir / filename)

                index += 1
                elapsed = time.time() - start_t
                _append_event(
                    args.event_log,
                    type="image",
                    index=index,
                    total=total,
                    filename=filename,
                    prompt=prompt,
                    prompt_index=pi,
                    seed=this_seed,
                    elapsed=elapsed,
                )

                if device == "cuda":
                    torch.cuda.empty_cache()

        _append_event(
            args.event_log,
            type="completed",
            total=index,
            elapsed=time.time() - start_t,
        )
        return 0
    except KeyboardInterrupt:
        _append_event(args.event_log, type="cancelled", reason="interrupted")
        return 130
    except Exception as e:
        _append_event(args.event_log, type="error", error=str(e), error_type=type(e).__name__)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
