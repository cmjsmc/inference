import base64
import gc
import io
import json
import logging
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen35ChatHandler


# Completely silence logging and output
logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.CRITICAL, handlers=[logging.NullHandler()])
logger = logging.getLogger("qwen36 studio")
logger.setLevel(logging.CRITICAL)
logger.propagate = False


@contextmanager
def suppress_stdout_stderr():
    """Safely redirects C/C++ engine file descriptors to /dev/null during inference."""
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_stdout = os.dup(1)
        save_stderr = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(save_stdout, 1)
            os.dup2(save_stderr, 2)
            os.close(null_fd)
            os.close(save_stdout)
            os.close(save_stderr)
        except Exception:
            pass


# ==============================================================================
# VISION-LANGUAGE MODEL CONFIGURATION
# ==============================================================================
MODEL_REPO = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
MODEL_FILE = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-F32.gguf"

llm_instance = None
llm_lock = threading.Lock()
session_keys: Dict[str, Tuple[ec.EllipticCurvePrivateKey, float]] = {}
tasks: Dict[str, dict] = {}

HKDF_SALT = b"qwen-vision-zero-leak-salt-2025"
HKDF_INFO = b"qwen-vision-e2ee"


# ==============================================================================
# WD TAGGER & OVERLAP FILTER CONFIGURATION
# ==============================================================================
WD_MODEL_REPO_MAP = {
    "eva02-canary": {
        "repo_id": "ashen-sensored/wd-eva02-tagger-2026-canary",
        "model_file": "model.safetensors",
        "tags_file": "selected_tags.csv",
        "arch": "eva02_large_patch14_448",
    },
    "vit": {
        "repo_id": "SmilingWolf/wd-vit-tagger-v3",
        "model_file": None,
        "tags_file": "selected_tags.csv",
        "arch": None,
    },
    "swinv2": {
        "repo_id": "SmilingWolf/wd-swinv2-tagger-v3",
        "model_file": None,
        "tags_file": "selected_tags.csv",
        "arch": None,
    },
    "convnext": {
        "repo_id": "SmilingWolf/wd-convnext-tagger-v3",
        "model_file": None,
        "tags_file": "selected_tags.csv",
        "arch": None,
    },
}

ACTIVE_WD_MODEL = "eva02-canary"
WD_GEN_THRESHOLD = 0.35
WD_CHAR_THRESHOLD = 0.75
WD_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANGE_MAP = [
    ["from behind", "shot from behind"],
    ["from below", "shot from a low angle"],
    ["from above", "shot from a high angle"],
    ["all fours", "on all fours"],
    ["1girl", "a female"],
    ["1boy", "a male"],

    ["oral", "oral sex"],
    ["vaginal", "vaginal sex"],
    ["anal", "anal sex"],
    ["sex", "having sex"],
    ["all fours", "on all fours"],

]

IMPORTANT_TAGS = [
    "POV", "shot from a low angle", "shot from a high angle", "shot from behind",
    "on all fours", "straddling",
    "squatting", "kneeling", "sitting", "standing", 
    "indoors", "outdoors",
    
    "anal penetration", "anal sex",
    "vaginal penetration", "vaginal sex",
    "oral sex", "blowjob", "deepthroat",
    "implied sex", "implied penetration",
    "penis", "testicles", "ass", "pussy",
    "nude",
]

wd_model_instance: Optional[nn.Module] = None
wd_labels_instance: Optional["LabelData"] = None
wd_transform_instance = None
overlap_dict_instance: Optional[dict[str, list[str]]] = None


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    """Converts in-RAM image to RGB/RGBA on a neutral white canvas."""
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    """Pads image to square with a white background entirely in RAM."""
    w, h = image.size
    px = max(image.size)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def load_labels_hf(repo_id: str, filename: str = "selected_tags.csv") -> LabelData:
    try:
        csv_path = hf_hub_download(repo_id=repo_id, filename=filename)
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"{filename} failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    return LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )


def load_overlap_dict() -> dict[str, list[str]]:
    """Loads tag overlap dictionary from Hugging Face cache into RAM."""
    try:
        json_file = hf_hub_download(
            "alea31415/tag_filtering",
            "overlap_tags_simplified.json",
            repo_type="dataset",
        )
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_wd_tagger(model_key: str = ACTIVE_WD_MODEL):
    """Pre-caches WD model weights, label mappings, transform, and overlap dict in memory with CUDA OOM safety."""
    global wd_model_instance, wd_labels_instance, wd_transform_instance, overlap_dict_instance, WD_DEVICE
    if wd_model_instance is None:
        target_cfg = WD_MODEL_REPO_MAP.get(model_key, WD_MODEL_REPO_MAP["eva02-canary"])
        repo_id = target_cfg["repo_id"]
        model_file = target_cfg["model_file"]
        tags_file = target_cfg["tags_file"]
        arch = target_cfg["arch"]

        try:
            model = timm.create_model("hf-hub:" + repo_id, pretrained=True).eval()
        except Exception:
            try:
                model = timm.create_model("hf-hub:" + repo_id, pretrained=False).eval()
                if model_file:
                    state_dict = timm.models.load_state_dict_from_hf(repo_id, filename=model_file)
                else:
                    state_dict = timm.models.load_state_dict_from_hf(repo_id)
                model.load_state_dict(state_dict)
            except Exception:
                arch_name = arch or "eva02_large_patch14_448"
                try:
                    model = timm.create_model(arch_name, pretrained=False, num_classes=16473).eval()
                except Exception:
                    model = timm.create_model("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", pretrained=False, num_classes=16473).eval()

                weights_path = hf_hub_download(repo_id=repo_id, filename=model_file or "model.safetensors")
                try:
                    from safetensors.torch import load_file as load_safetensors
                    state_dict = load_safetensors(weights_path)
                except ImportError:
                    state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)

        # Enforce exact EVA02 / WD V3 data transform (Bicubic, 448x448, crop_pct=1.0, mean/std=0.5)
        wd_transform_instance = create_transform(
            input_size=448,
            interpolation="bicubic",
            crop_pct=1.0,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

        # Attempt to load onto GPU in FP16 to halve VRAM; fallback gracefully to CPU on OOM
        if WD_DEVICE.type == "cuda":
            try:
                wd_model_instance = model.half().to(WD_DEVICE)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                WD_DEVICE = torch.device("cpu")
                wd_model_instance = model.float().to("cpu")
        else:
            wd_model_instance = model.float().to("cpu")

        wd_labels_instance = load_labels_hf(repo_id=repo_id, filename=tags_file)
        overlap_dict_instance = load_overlap_dict()

    return wd_model_instance, wd_labels_instance, wd_transform_instance, overlap_dict_instance


def get_raw_tags(probs: Tensor, labels: LabelData, gen_threshold: float, char_threshold: float) -> list[str]:
    """Sorts and extracts prediction tags by confidence threshold."""
    probs_list = list(zip(labels.names, probs.numpy()))

    gen_labels = [probs_list[i] for i in labels.general if probs_list[i][1] > gen_threshold]
    gen_labels.sort(key=lambda item: item[1], reverse=True)

    char_labels = [probs_list[i] for i in labels.character if probs_list[i][1] > char_threshold]
    char_labels.sort(key=lambda item: item[1], reverse=True)

    combined_names = [x[0] for x in gen_labels] + [x[0] for x in char_labels]
    return combined_names


def filter_overlap_tags(tags: list[str], overlap_dict: dict[str, list[str]], check_superwords: bool = True) -> list[str]:
    """Filters redundant overlap tags and superwords."""
    if not tags:
        return []

    tags_underscore = [t.replace(" ", "_") for t in tags]
    tags_underscore_set = set(tags_underscore)
    result: list[str] = []

    for tag, tag_ in zip(tags, tags_underscore):
        to_remove = False

        if tag_ in overlap_dict:
            overlap_values = set(overlap_dict[tag_])
            if overlap_values.intersection(tags_underscore_set):
                to_remove = True

        if not to_remove and check_superwords:
            for tag_another_ in tags_underscore:
                if tag_ != tag_another_ and tag_ in tag_another_:
                    to_remove = True
                    break

        if not to_remove:
            result.append(tag)

    return result


def process_tags_pipeline(image_data: bytearray) -> tuple[str, str]:
    """Executes WD model inference, removes overlaps, applies CHANGE_MAP, and detects IMPORTANT_TAGS in RAM."""
    global WD_DEVICE
    wd_model, labels, transform, overlap_dict = load_wd_tagger()

    with Image.open(io.BytesIO(image_data)) as pil_img:
        rgb_img = pil_ensure_rgb(pil_img)
        padded_img = pil_pad_square(rgb_img)
        inputs: Tensor = transform(padded_img).unsqueeze(0)[:, [2, 1, 0]]
        del rgb_img, padded_img

    if WD_DEVICE.type == "cuda":
        inputs = inputs.half().to(WD_DEVICE)
    else:
        inputs = inputs.float().to(WD_DEVICE)

    with suppress_stdout_stderr(), torch.inference_mode():
        try:
            outputs = wd_model(inputs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            # Fallback to CPU on live VRAM contention
            WD_DEVICE = torch.device("cpu")
            wd_model = wd_model.float().to("cpu")
            inputs = inputs.float().to("cpu")
            outputs = wd_model(inputs)

        outputs = F.sigmoid(outputs)
        probs = outputs.squeeze(0).float().cpu()

    # 1. Tag Inference
    raw_tags = get_raw_tags(probs, labels, WD_GEN_THRESHOLD, WD_CHAR_THRESHOLD)

    # 2. Remove Overlap
    filtered_tags = filter_overlap_tags(raw_tags, overlap_dict or {}, check_superwords=True)
    tags_spaced = [t.replace("_", " ") for t in filtered_tags]

    # 3. Apply CHANGE_MAP
    change_map_dict = {src.lower(): dst for src, dst in CHANGE_MAP}
    edited_tags = [change_map_dict.get(t.lower(), t) for t in tags_spaced]

    # 4. Check IMPORTANT_TAGS
    edited_tags_lower = {t.lower() for t in edited_tags}
    high_tags_present = [imp for imp in IMPORTANT_TAGS if imp.lower() in edited_tags_lower]

    tags_present = ", ".join(edited_tags)
    high_tags_present_str = ", ".join(high_tags_present)

    del inputs, outputs, probs, raw_tags, filtered_tags, tags_spaced, edited_tags, edited_tags_lower
    return tags_present, high_tags_present_str


def zero_mem(target):
    """Wipes mutable memory buffers with zeros."""
    if isinstance(target, (bytearray, memoryview)):
        for i in range(len(target)):
            target[i] = 0


def sanitize_img_data_uri(image_str: str) -> str:
    """Normalizes any base64 image data URI to strict RFC-compliant format."""
    if not image_str:
        return ""
    if "," in image_str:
        _, b64_data = image_str.split(",", 1)
    else:
        b64_data = image_str

    clean_b64 = re.sub(r"[^A-Za-z0-9+/=]", "", b64_data)
    return f"data:image/jpeg;base64,{clean_b64}"


def load_model():
    global llm_instance
    if llm_instance is None:
        try:
            chat_handler = Qwen35ChatHandler.from_pretrained(
                repo_id=MODEL_REPO,
                filename=MMPROJ_FILE,
                image_min_tokens=1024,
                enable_thinking=False,
                verbose=False,
            )
        except Exception:
            chat_handler = None

        llm_instance = Llama.from_pretrained(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            chat_handler=chat_handler,
            chat_format="qwen3.6",
            n_gpu_layers="auto",
            tensor_split=[0.5, 0.5],
            n_ctx=8192,
            n_threads=2,
            n_threads_batch=2,
            n_batch=4096,
            n_ubatch=1024,
            n_seq_max=2,
            flash_attn=False,
            verbose=False,
            verbosity=0,
            load_mode=llama_cpp.llama_load_mode.LLAMA_LOAD_MODE_MMAP_MLOCK,
        )
    return llm_instance


def cleanup_expired_records():
    """Purges expired sessions and orphaned task results from RAM."""
    now = time.time()
    expired_sessions = [k for k, (_, t) in session_keys.items() if now - t > 300]
    for k in expired_sessions:
        del session_keys[k]

    expired_tasks = [k for k, v in tasks.items() if now - v.get("created_at", 0) > 300]
    for k in expired_tasks:
        del tasks[k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_wd_tagger()
    yield
    global llm_instance, wd_model_instance, wd_labels_instance, wd_transform_instance, overlap_dict_instance, session_keys, tasks
    session_keys.clear()
    tasks.clear()
    del llm_instance
    del wd_model_instance
    del wd_labels_instance
    del wd_transform_instance
    del overlap_dict_instance
    gc.collect()


app = FastAPI(title="Qwen 3.6 Private Studio", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def process_inference_task(task_id: str, server_priv: ec.EllipticCurvePrivateKey, payload: "EncryptedPayload"):
    try:
        # 1. Derive Shared Key
        client_pub_bytes = base64.b64decode(payload.client_public_key)
        client_pub = serialization.load_der_public_key(client_pub_bytes)
        shared_secret = server_priv.exchange(ec.ECDH(), client_pub)

        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=HKDF_SALT,
            info=HKDF_INFO,
        ).derive(shared_secret)

        aesgcm = AESGCM(derived_key)

        # 2. Decrypt In-RAM
        iv_bytes = base64.b64decode(payload.iv)
        ciphertext_bytes = base64.b64decode(payload.ciphertext)
        decrypted_raw = aesgcm.decrypt(iv_bytes, ciphertext_bytes, None)

        decrypted_buf = bytearray(decrypted_raw)
        del decrypted_raw

        decrypted_dict = json.loads(decrypted_buf.decode("utf-8"))
        zero_mem(decrypted_buf)
        del decrypted_buf

        prompt_str = decrypted_dict.get("prompt", "")
        raw_image_str = decrypted_dict.get("image", "")
        image_str = sanitize_img_data_uri(raw_image_str)
        del raw_image_str

        max_tokens = int(decrypted_dict.get("max_tokens", 512))
        temperature = float(decrypted_dict.get("temperature", 0.2))
        del decrypted_dict

        # 3. Perform Tag Extraction, Overlap Removal, and Prompt Formatting In-RAM
        b64_part = image_str.split(",", 1)[1] if "," in image_str else image_str
        image_buf = bytearray(base64.b64decode(b64_part))

        try:
            tags_present, high_tags_present = process_tags_pipeline(image_buf)
        finally:
            # Strictly wipe the decoded image buffer even if tagging fails
            zero_mem(image_buf)
            del image_buf

        if tags_present:
            tags_prompt = f"Describe the image through concise, clear sentences, making use of the following keywords, if relevant: [{tags_present}]"
            if high_tags_present:
                high_impact_prompt = f"{tags_prompt} You must include the following keywords: [{high_tags_present}]"
                final_prompt = f"{prompt_str}\n\n{high_impact_prompt}".strip() if prompt_str else high_impact_prompt
            else:
                final_prompt = f"{prompt_str}\n\n{tags_prompt}".strip() if prompt_str else tags_prompt
        else:
            final_prompt = prompt_str

        # 4. Model Execution (Thread-Safe)
        model = load_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt},
                    {"type": "image_url", "image_url": {"url": image_str}},
                ],
            }
        ]
        del prompt_str
        del final_prompt
        del image_str

        with llm_lock, suppress_stdout_stderr():
            model.reset()
            response = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.80,
                min_p=0.05,
                present_penalty=0.01,
                repeat_penalty=1.01,
                penalty_last_n=64,
            )
        del messages

        content_str = response["choices"][0]["message"]["content"]
        del response

        # 5. Re-Encrypt Response
        resp_bytes = bytearray(json.dumps({"content": content_str}).encode("utf-8"))
        del content_str

        out_iv = os.urandom(12)
        encrypted_out = aesgcm.encrypt(out_iv, bytes(resp_bytes), None)
        zero_mem(resp_bytes)
        del resp_bytes

        del aesgcm
        del derived_key
        del shared_secret
        del server_priv
        gc.collect()

        tasks[task_id] = {
            "status": "completed",
            "data": {
                "iv": base64.b64encode(out_iv).decode("utf-8"),
                "ciphertext": base64.b64encode(encrypted_out).decode("utf-8"),
            },
            "created_at": time.time(),
        }

    except Exception as e:
        gc.collect()
        tasks[task_id] = {
            "status": "failed",
            "error": f"Inference execution error: {str(e)}",
            "created_at": time.time(),
        }


class EncryptedPayload(BaseModel):
    session_id: str
    client_public_key: str
    iv: str
    ciphertext: str


@app.get("/api/handshake")
def handshake():
    cleanup_expired_records()

    server_priv = ec.generate_private_key(ec.SECP256R1())
    server_pub = server_priv.public_key()
    session_id = base64.b64encode(os.urandom(16)).decode("utf-8")
    session_keys[session_id] = (server_priv, time.time())

    pub_der = server_pub.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return {
        "session_id": session_id,
        "server_public_key": base64.b64encode(pub_der).decode("utf-8"),
    }


@app.post("/api/generate")
def generate_encrypted(payload: EncryptedPayload):
    cleanup_expired_records()

    if payload.session_id not in session_keys:
        raise HTTPException(status_code=400, detail="Session expired or invalid. Please retry.")

    server_priv, _ = session_keys.pop(payload.session_id)
    task_id = base64.b64encode(os.urandom(16)).decode("utf-8")

    tasks[task_id] = {"status": "processing", "created_at": time.time()}

    worker_thread = threading.Thread(
        target=process_inference_task,
        args=(task_id, server_priv, payload),
        daemon=True,
    )
    worker_thread.start()

    return {"task_id": task_id}


@app.get("/api/status/{task_id}")
def check_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found or expired.")

    task = tasks[task_id]
    if task["status"] == "completed":
        res = tasks.pop(task_id)
        return {"status": "completed", "data": res["data"]}
    elif task["status"] == "failed":
        res = tasks.pop(task_id)
        return {"status": "failed", "error": res.get("error", "Unknown error")}
    else:
        return {"status": "processing"}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen 3.6 Private Studio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: { 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca' }
                    }
                }
            }
        }
    </script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        .prose pre { background: #0f172a; padding: 1rem; border-radius: 0.5rem; }
        .dropzone-active { border-color: #6366f1 !important; background-color: rgba(99, 102, 241, 0.08) !important; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans selection:bg-brand-500 selection:text-white">

    <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="p-2 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/20">
                    <i data-lucide="shield-check" class="w-5 h-5"></i>
                </div>
                <div>
                    <h1 class="font-bold text-lg leading-tight tracking-tight">Qwen 3.6 Private Studio</h1>
                    <p class="text-xs text-slate-400">Zero Disk Logging • In-RAM Execution • In-Browser E2EE</p>
                </div>
            </div>
            <div class="flex items-center space-x-2 text-xs text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-3 py-1.5 rounded-full font-medium">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span>E2EE Active (AES-256-GCM)</span>
            </div>
        </div>
    </header>

    <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

        <!-- Left: Input Controls -->
        <div class="lg:col-span-6 flex flex-col gap-6">

            <div class="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 backdrop-blur shadow-sm">
                <div class="flex items-center justify-between mb-3">
                    <label class="font-semibold text-sm flex items-center gap-2 text-slate-200">
                        <i data-lucide="image" class="w-4 h-4 text-brand-500"></i> Image Input
                    </label>
                    <button id="clearImageBtn" class="hidden text-xs text-rose-400 hover:text-rose-300 transition flex items-center gap-1">
                        <i data-lucide="trash-2" class="w-3.5 h-3.5"></i> Clear
                    </button>
                </div>

                <div id="dropzone" class="relative border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-xl transition cursor-pointer overflow-hidden min-h-[260px] flex flex-col items-center justify-center bg-slate-950/40 group">
                    <input type="file" id="fileInput" accept="image/*" class="hidden" />

                    <div id="dropzonePlaceholder" class="p-8 text-center flex flex-col items-center">
                        <div class="w-14 h-14 rounded-full bg-slate-800/80 border border-slate-700 flex items-center justify-center text-slate-400 group-hover:scale-110 group-hover:text-brand-500 group-hover:border-brand-500/50 transition duration-200 mb-3">
                            <i data-lucide="upload-cloud" class="w-7 h-7"></i>
                        </div>
                        <p class="text-sm font-medium text-slate-300">Drag & drop image here, or <span class="text-brand-500 underline underline-offset-4">browse</span></p>
                        <p class="text-xs text-slate-500 mt-1">Automatically sanitizes filename to <code class="text-emerald-400">image_01.jpg</code></p>
                    </div>

                    <div id="previewContainer" class="hidden relative w-full h-full min-h-[260px] flex items-center justify-center bg-slate-950 p-2">
                        <img id="imagePreview" src="" alt="Preview" class="max-h-[360px] w-auto max-w-full rounded-lg object-contain" />
                        <div class="absolute inset-0 bg-black/50 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center text-xs font-semibold text-white backdrop-blur-[2px]">
                            Drop new image or click to replace
                        </div>
                    </div>
                </div>
            </div>

            <div class="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 backdrop-blur shadow-sm flex flex-col gap-3">
                <div class="flex items-center justify-between">
                    <label class="font-semibold text-sm flex items-center gap-2 text-slate-200">
                        <i data-lucide="lock" class="w-4 h-4 text-emerald-400"></i> Prompt Instruction
                    </label>
                    <button id="resetPromptBtn" class="text-xs text-slate-400 hover:text-slate-200 transition">Reset default</button>
                </div>
                <textarea id="promptInput" rows="4" class="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl p-3.5 text-sm text-slate-200 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 resize-y" placeholder="Enter prompt..."></textarea>

                <div class="grid grid-cols-2 gap-4 pt-2 border-t border-slate-800/80">
                    <div>
                        <div class="flex justify-between text-xs text-slate-400 mb-1">
                            <span>Temperature</span>
                            <span id="tempVal" class="font-mono text-slate-300">0.2</span>
                        </div>
                        <input type="range" id="tempInput" min="0" max="1" step="0.05" value="0.2" class="w-full accent-brand-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg appearance-none">
                    </div>
                    <div>
                        <div class="flex justify-between text-xs text-slate-400 mb-1">
                            <span>Max Tokens</span>
                            <span id="tokensVal" class="font-mono text-slate-300">512</span>
                        </div>
                        <input type="range" id="tokensInput" min="64" max="2048" step="64" value="512" class="w-full accent-brand-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg appearance-none">
                    </div>
                </div>

                <button id="submitBtn" class="mt-2 w-full py-3.5 px-4 bg-brand-600 hover:bg-brand-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-medium rounded-xl transition flex items-center justify-center gap-2 shadow-lg shadow-brand-500/20">
                    <i data-lucide="lock" class="w-4 h-4"></i>
                    <span id="submitBtnText">Encrypt & Run Analysis</span>
                </button>
            </div>
        </div>

        <!-- Right: Output Display -->
        <div class="lg:col-span-6 flex flex-col">
            <div class="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 backdrop-blur shadow-sm flex-1 flex flex-col min-h-[500px]">

                <div class="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
                    <span class="font-semibold text-sm flex items-center gap-2 text-slate-200">
                        <i data-lucide="terminal" class="w-4 h-4 text-emerald-400"></i> Decrypted Output
                    </span>

                    <button id="copyBtn" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium transition flex items-center gap-1.5 border border-slate-700/60">
                        <i data-lucide="copy" class="w-3.5 h-3.5"></i>
                        <span id="copyText">Copy Markdown</span>
                    </button>
                </div>

                <div id="outputContainer" class="flex-1 overflow-y-auto">
                    <div id="emptyState" class="h-full flex flex-col items-center justify-center text-slate-500 py-16">
                        <i data-lucide="shield" class="w-12 h-12 mb-3 text-slate-700"></i>
                        <p class="text-sm">End-to-End Encrypted environment ready.</p>
                        <p class="text-xs text-slate-600 mt-1">Upload an image and click generate to begin.</p>
                    </div>

                    <div id="loadingState" class="hidden h-full flex flex-col items-center justify-center py-16 gap-3">
                        <div class="w-8 h-8 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin"></div>
                        <p id="loadingStatusText" class="text-sm text-slate-400 animate-pulse">Running In-RAM Inference...</p>
                    </div>

                    <div id="markdownOutput" class="hidden prose prose-invert max-w-none text-slate-200 text-sm leading-relaxed"></div>
                </div>

            </div>
        </div>

    </main>

    <script>
        lucide.createIcons();

        const DEFAULT_PROMPT = "Describe this image in concise sentences. The written description will be given to a photographer so they can reproduce the image. The description should clearly describe all the relevant elements of the image.";
        const HKDF_SALT = new TextEncoder().encode("qwen-vision-zero-leak-salt-2025");
        const HKDF_INFO = new TextEncoder().encode("qwen-vision-e2ee");

        let currentBase64Image = null;
        let rawOutputMarkdown = "";

        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const dropzonePlaceholder = document.getElementById('dropzonePlaceholder');
        const previewContainer = document.getElementById('previewContainer');
        const imagePreview = document.getElementById('imagePreview');
        const clearImageBtn = document.getElementById('clearImageBtn');

        const promptInput = document.getElementById('promptInput');
        const resetPromptBtn = document.getElementById('resetPromptBtn');
        const tempInput = document.getElementById('tempInput');
        const tempVal = document.getElementById('tempVal');
        const tokensInput = document.getElementById('tokensInput');
        const tokensVal = document.getElementById('tokensVal');

        const submitBtn = document.getElementById('submitBtn');
        const submitBtnText = document.getElementById('submitBtnText');
        const copyBtn = document.getElementById('copyBtn');
        const copyText = document.getElementById('copyText');

        const emptyState = document.getElementById('emptyState');
        const loadingState = document.getElementById('loadingState');
        const loadingStatusText = document.getElementById('loadingStatusText');
        const markdownOutput = document.getElementById('markdownOutput');

        promptInput.value = DEFAULT_PROMPT;
        tempInput.addEventListener('input', (e) => tempVal.textContent = e.target.value);
        tokensInput.addEventListener('input', (e) => tokensVal.textContent = e.target.value);
        resetPromptBtn.addEventListener('click', () => promptInput.value = DEFAULT_PROMPT);

        function arrayBufferToBase64(buffer) {
            let binary = '';
            const bytes = new Uint8Array(buffer);
            for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
            return window.btoa(binary);
        }

        function base64ToArrayBuffer(base64) {
            const binary = window.atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            return bytes.buffer;
        }

        async function parseResponse(response) {
            const text = await response.text();
            let json;
            try {
                json = JSON.parse(text);
            } catch (e) {
                throw new Error(`Server returned HTTP ${response.status}: ${text.slice(0, 150)}`);
            }
            if (!response.ok) {
                throw new Error(json.detail || `Request failed with status ${response.status}`);
            }
            return json;
        }

        function processAndSanitizeImage(rawFile, maxDimension = 1024) {
            return new Promise((resolve, reject) => {
                const sanitizedFile = new File([rawFile], "image_01.jpg", {
                    type: rawFile.type || "image/jpeg",
                    lastModified: Date.now()
                });

                const blobUrl = URL.createObjectURL(sanitizedFile);
                const img = new Image();

                img.onload = () => {
                    URL.revokeObjectURL(blobUrl);
                    let { width, height } = img;
                    if (width > maxDimension || height > maxDimension) {
                        if (width > height) {
                            height = Math.round((height * maxDimension) / width);
                            width = maxDimension;
                        } else {
                            width = Math.round((width * maxDimension) / height);
                            height = maxDimension;
                        }
                    }
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, width, height);

                    const cleanDataUri = canvas.toDataURL('image/jpeg', 0.85);
                    resolve(cleanDataUri);
                };

                img.onerror = () => {
                    URL.revokeObjectURL(blobUrl);
                    reject(new Error('Unsupported or corrupted image format.'));
                };

                img.src = blobUrl;
            });
        }

        dropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
        });

        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dropzone-active');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dropzone-active');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        });

        window.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let item of items) {
                if (item.kind === 'file' && item.type.startsWith('image/')) {
                    handleFile(item.getAsFile());
                    break;
                }
            }
        });

        async function handleFile(file) {
            try {
                currentBase64Image = await processAndSanitizeImage(file);
                imagePreview.src = currentBase64Image;
                dropzonePlaceholder.classList.add('hidden');
                previewContainer.classList.remove('hidden');
                clearImageBtn.classList.remove('hidden');
            } catch (e) {
                alert('Could not process image: ' + e.message);
            }
        }

        clearImageBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentBase64Image = null;
            imagePreview.src = "";
            fileInput.value = "";
            previewContainer.classList.add('hidden');
            dropzonePlaceholder.classList.remove('hidden');
            clearImageBtn.classList.add('hidden');
        });

        submitBtn.addEventListener('click', async () => {
            if (!currentBase64Image) {
                alert('Please select or drop an image first.');
                return;
            }
            if (!promptInput.value.trim()) {
                alert('Please provide a prompt.');
                return;
            }

            submitBtn.disabled = true;
            submitBtnText.textContent = "Processing...";
            emptyState.classList.add('hidden');
            markdownOutput.classList.add('hidden');
            loadingState.classList.remove('hidden');
            loadingStatusText.textContent = "Establishing E2EE Handshake...";

            try {
                const hsResp = await fetch('/api/handshake');
                const { session_id, server_public_key } = await parseResponse(hsResp);

                const clientKeyPair = await window.crypto.subtle.generateKey(
                    { name: "ECDH", namedCurve: "P-256" },
                    false,
                    ["deriveKey", "deriveBits"]
                );

                const clientSpki = await window.crypto.subtle.exportKey("spki", clientKeyPair.publicKey);
                const clientPubBase64 = arrayBufferToBase64(clientSpki);

                const serverKeyBuf = base64ToArrayBuffer(server_public_key);
                const serverPubKey = await window.crypto.subtle.importKey(
                    "spki",
                    serverKeyBuf,
                    { name: "ECDH", namedCurve: "P-256" },
                    false,
                    []
                );

                const sharedSecretBits = await window.crypto.subtle.deriveBits(
                    { name: "ECDH", public: serverPubKey },
                    clientKeyPair.privateKey,
                    256
                );

                const hkdfKey = await window.crypto.subtle.importKey(
                    "raw",
                    sharedSecretBits,
                    { name: "HKDF" },
                    false,
                    ["deriveKey"]
                );

                const aesKey = await window.crypto.subtle.deriveKey(
                    {
                        name: "HKDF",
                        hash: "SHA-256",
                        salt: HKDF_SALT,
                        info: HKDF_INFO
                    },
                    hkdfKey,
                    { name: "AES-GCM", length: 256 },
                    false,
                    ["encrypt", "decrypt"]
                );

                const iv = window.crypto.getRandomValues(new Uint8Array(12));
                const payloadData = JSON.stringify({
                    prompt: promptInput.value,
                    image: currentBase64Image,
                    temperature: parseFloat(tempInput.value),
                    max_tokens: parseInt(tokensInput.value),
                });

                const payloadBytes = new TextEncoder().encode(payloadData);
                const encryptedPayload = await window.crypto.subtle.encrypt(
                    { name: "AES-GCM", iv: iv },
                    aesKey,
                    payloadBytes
                );

                loadingStatusText.textContent = "Encrypting & Queuing Task...";
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: session_id,
                        client_public_key: clientPubBase64,
                        iv: arrayBufferToBase64(iv),
                        ciphertext: arrayBufferToBase64(encryptedPayload)
                    })
                });

                const { task_id } = await parseResponse(response);

                let encResponse = null;
                let elapsedSeconds = 0;

                while (!encResponse) {
                    await new Promise(r => setTimeout(r, 1500));
                    elapsedSeconds += 1.5;
                    loadingStatusText.textContent = `Running In-RAM Inference... (${Math.round(elapsedSeconds)}s)`;

                    const statusResp = await fetch(`/api/status/${task_id}`);
                    const statusData = await parseResponse(statusResp);

                    if (statusData.status === 'completed') {
                        encResponse = statusData.data;
                    } else if (statusData.status === 'failed') {
                        throw new Error(statusData.error || 'Inference execution failed on server.');
                    }
                }

                loadingStatusText.textContent = "Decrypting Output in Browser...";
                const respIv = base64ToArrayBuffer(encResponse.iv);
                const respCiphertext = base64ToArrayBuffer(encResponse.ciphertext);

                const decryptedBytes = await window.crypto.subtle.decrypt(
                    { name: "AES-GCM", iv: new Uint8Array(respIv) },
                    aesKey,
                    respCiphertext
                );

                const decryptedJson = JSON.parse(new TextDecoder().decode(decryptedBytes));
                rawOutputMarkdown = decryptedJson.content;

                markdownOutput.innerHTML = marked.parse(rawOutputMarkdown);
                markdownOutput.querySelectorAll('pre code').forEach((el) => {
                    hljs.highlightElement(el);
                });

                loadingState.classList.add('hidden');
                markdownOutput.classList.remove('hidden');

            } catch (err) {
                loadingState.classList.add('hidden');
                emptyState.classList.remove('hidden');
                alert(err.message);
            } finally {
                submitBtn.disabled = false;
                submitBtnText.textContent = "Encrypt & Run Analysis";
            }
        });

        copyBtn.addEventListener('click', async () => {
            if (!rawOutputMarkdown) return;
            try {
                await navigator.clipboard.writeText(rawOutputMarkdown);
                copyText.textContent = "Copied!";
                setTimeout(() => copyText.textContent = "Copy Markdown", 2000);
            } catch (err) {
                alert('Failed to copy to clipboard');
            }
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(content=HTML_TEMPLATE)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False,
        access_log=False,
        log_level="critical",
        timeout_keep_alive=180,
    )
