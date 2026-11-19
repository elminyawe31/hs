# syntax=docker/dockerfile:1
# Railway: create a project from a GitHub repository containing this Dockerfile;
# deploy it as a Dockerfile service. Railway supplies PORT automatically.
# CPU only; no paid API and no runtime model downloads.
# Experimental: live challenge solving and Docker/Railway builds are unverified.
# Local checks: model downloads, hashes, CPU dummy inferences and mocked-browser
# HTTP /health and /solve passed. The returned mock token was not a live token.
# All ONNX release assets are downloaded at build time. Filename routing is a
# heuristic; browser selectors and actual hCaptcha behaviour are unverified.
# Models that fail validation are skipped; the build fails if none pass.
# The service has no authentication and accepts arbitrary URLs and proxies;
# do not expose it to untrusted users or networks without access controls.
# Vendored ONNX code: QIN2DIM/hcaptcha-challenger archive, revision
# 28ee79b80bad2e5a7d713acdf2d9376e3a46e73e, GPL-3.0-or-later.
# The bee ONNX metadata identifies Ultralytics and an AGPL-3.0 license;
# check upstream model licenses before redistribution.
FROM python:3.11-slim
ENV OMP_NUM_THREADS=2 ORT_DISABLE_TELEMETRY=1 PORT=8000 MODEL_DIR=/app/models PYTHONUNBUFFERED=1 USE_TORCH=0 USE_TF=0
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir 'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' 'onnxruntime>=1.18,<2' 'opencv-python-headless>=4.10,<5' 'numpy>=1.26,<3' 'scipy>=1.11,<2' 'pillow>=10.4,<13' 'loguru>=0.7,<1' 'requests>=2.32,<3' 'playwright>=1.44,<2' 'importlib-metadata>=7,<9'
RUN playwright install --with-deps firefox
RUN mkdir -p /app/vendor/archived_onnx /app/models
RUN cat <<'ARENA_FILE_END' > /app/main.py
"""FastAPI interface for the best-effort ONNX browser solver."""
import logging
import os
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from solver import HCaptchaSolver, ModelUnavailable, SolveFailure, UnsupportedChallenge

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
app = FastAPI()
MODEL_DIR = Path(os.getenv('MODEL_DIR', '/app/models'))

class SolveRequest(BaseModel):
    url: HttpUrl
    sitekey: str | None = None
    proxy: str | None = None

@app.get('/health')
def health():
    count = len(list(MODEL_DIR.glob('*.onnx'))) if MODEL_DIR.is_dir() else 0
    ready = count > 0 and (MODEL_DIR / 'registry.json').is_file()
    return {'status': 'ok' if ready else 'not_ready', 'models_loaded': False, 'model_files': count, 'registry_present': (MODEL_DIR / 'registry.json').is_file()}

@app.post('/solve')
async def solve(request: SolveRequest):
    start = time.monotonic()
    try:
        solver = HCaptchaSolver(MODEL_DIR, proxy=request.proxy)
        token = await solver.solve(str(request.url), request.sitekey)
        elapsed = round((time.monotonic() - start)*1000)
        log.info('solve completed in %d ms', elapsed)
        return {'token': token, 'elapsed_ms': elapsed, 'status': 'success'}
    except ModelUnavailable as exc: raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (UnsupportedChallenge, ValueError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception('solve failed')
        raise HTTPException(status_code=500, detail=f'Solve failed: {exc}') from exc
ARENA_FILE_END
RUN cat <<'ARENA_FILE_END' > /app/solver.py
"""Best-effort Firefox/ONNX challenge driver. No external vision service is used.

Selectors and filename-based routing describe an unverified upstream UI, not a
contract. Unsupported or ambiguous challenges fail rather than guess.
"""
import asyncio
import ast
import json
import logging
import math
import random
import re
import threading
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import onnxruntime as ort
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from vendor.archived_onnx.resnet import ResNetControl
from vendor.archived_onnx.yolo import YOLOv8

log = logging.getLogger(__name__)

class ModelUnavailable(Exception): pass
class UnsupportedChallenge(Exception): pass
class SolveFailure(Exception): pass

class ORTNet:
    def __init__(self, path: str):
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)':
            raise ValueError('Expected one float input')
        self.input_name = inputs[0].name
        self.blob = None
    def setInput(self, blob): self.blob = np.asarray(blob, dtype=np.float32)
    def forward(self): return self.session.run(None, {self.input_name: self.blob})[0]

@lru_cache(maxsize=16)
def load_binary(path: str): return ResNetControl.from_pluggable_model(ORTNet(path))

# Archived ResNet control mutates its DNN net input; protect shared cached sessions.
_binary_lock = threading.Lock()
def classify_binary(model: Path, image: bytes) -> bool:
    if not model.is_file(): raise ModelUnavailable(f'Missing model: {model}')
    with _binary_lock:
        return bool(load_binary(str(model.resolve())).execute(image))

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', text.lower())).strip()

def subject(prompt: str) -> str:
    text = normalize(prompt)
    text = re.sub(r'^(?:please )?(?:select|click|choose|identify|mark)(?: on)? (?:all |the )?(?:images|pictures|tiles|squares)(?: that)? (?:contain|containing|with|of|showing) (?:all |the |a |an )?', '', text)
    text = re.sub(r'^(?:please )?(?:click|select|choose|mark)(?: on)? (?:all |the |a |an )?', '', text)
    return text.strip()

from registry_builder import build_registry

class Registry:
    def __init__(self, directory: Path):
        self.directory = directory
        path = directory / 'registry.json'
        if not path.is_file(): raise ModelUnavailable(f'Missing registry: {path}')
        self.entries = json.loads(path.read_text())['entries']
    def resolve(self, prompt, kind):
        target = subject(prompt)
        matches = [e for e in self.entries if e['kind'] == kind and e['label'] == target]
        # Prefer an unversioned exact filename; otherwise reject ambiguous versions.
        preferred = [e for e in matches if Path(e['model']).stem.lower() == target.replace(' ', '_')]
        if len(preferred) == 1: matches = preferred
        if not matches: raise ModelUnavailable(f'No {kind} model registered for prompt {prompt!r} (subject {target!r})')
        if len(matches) != 1: raise UnsupportedChallenge(f'Ambiguous models for {target!r}: {[e["model"] for e in matches]}')
        filename = matches[0]['model']
        if Path(filename).name != filename: raise UnsupportedChallenge('Unsafe model filename in registry')
        path = self.directory / filename
        if not path.is_file(): raise ModelUnavailable(f'Missing model: {path}')
        return path

def detect_yolo(model: Path, image: bytes, target: str):
    """Use archived YOLOv8 preprocessing + multiclass NMS; return image-space centres."""
    if not model.is_file(): raise ModelUnavailable(f'Missing model: {model}')
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model), sess_options=opts, providers=['CPUExecutionProvider'])
    metadata = session.get_modelmeta().custom_metadata_map
    try: names = ast.literal_eval(metadata['names'])
    except (KeyError, SyntaxError, ValueError) as e:
        raise UnsupportedChallenge(f'{model.name} has no readable YOLO class names') from e
    if isinstance(names, dict):
        if set(names) != set(range(len(names))): raise UnsupportedChallenge('Non-contiguous YOLO class IDs')
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise UnsupportedChallenge('Unsupported YOLO class metadata')
    # Exact class matching avoids selecting unrelated classes from broad detectors.
    match = normalize(target)
    matching = {i for i, name in enumerate(names) if normalize(name) == match}
    if not matching: raise UnsupportedChallenge(f'YOLO classes {names!r} do not contain {target!r}')
    detector = YOLOv8.from_pluggable_model(session, classes=names)
    result = detector(image, shape_type='point')
    return [(int(x), int(y)) for label, (x, y), score in result if normalize(label) == match]

async def human_click(page, box, point=None):
    if not box: raise SolveFailure('Element is not visible')
    x = box['x'] + (point[0] if point else box['width'] / 2)
    y = box['y'] + (point[1] if point else box['height'] / 2)
    if not (box['x'] <= x <= box['x'] + box['width'] and box['y'] <= y <= box['y'] + box['height']):
        raise SolveFailure('Detection lies outside the image')
    sx, sy = max(0, x - random.uniform(30, 100)), max(0, y - random.uniform(30, 100))
    cx, cy = (sx + x) / 2 + random.uniform(-20, 20), (sy + y) / 2 + random.uniform(-20, 20)
    for i in range(1, 13):
        t = i / 12; t = t * t * (3 - 2 * t)
        await page.mouse.move((1-t)**2*sx + 2*(1-t)*t*cx + t*t*x, (1-t)**2*sy + 2*(1-t)*t*cy + t*t*y)
        await asyncio.sleep(0.005)
    await page.mouse.click(x, y, delay=random.randint(40, 100))

class BrowserChallenge:
    """DOM-based best-effort browser driver. All selectors need live validation."""
    def __init__(self, page): self.page = page
    async def token(self):
        return await self.page.evaluate('''() => { const e = document.querySelector('textarea[name="h-captcha-response"], input[name="h-captcha-response"]'); return e ? e.value : ''; }''')
    async def start(self):
        checkbox = self.page.frame_locator('iframe[title*="checkbox for hCaptcha"]').locator('#checkbox')
        await checkbox.wait_for(state='visible', timeout=20000)
        await human_click(self.page, await checkbox.bounding_box())
    async def wait_challenge(self):
        iframe = self.page.locator('iframe[title*="hCaptcha challenge"]')
        await iframe.wait_for(state='visible', timeout=20000)
        frame = self.page.frame_locator('iframe[title*="hCaptcha challenge"]')
        header = frame.locator('.prompt-text, .challenge-header .prompt-text')
        await header.first.wait_for(state='visible', timeout=10000)
        prompt = (await header.first.inner_text()).strip()
        if not prompt: raise UnsupportedChallenge('Challenge prompt is empty')
        tiles = frame.locator('.task-image')
        if await tiles.count(): return frame, prompt, 'binary', tiles
        canvas = frame.locator('.challenge-view canvas')
        if await canvas.count(): return frame, prompt, 'area_select', canvas
        raise UnsupportedChallenge(f'Unsupported challenge layout: {prompt}')
    async def run_round(self, registry):
        frame, prompt, kind, images = await self.wait_challenge()
        model = registry.resolve(prompt, kind)
        log.info('challenge kind=%s model=%s', kind, model.name)
        if kind == 'binary':
            for i in range(await images.count()):
                tile = images.nth(i)
                image = await tile.screenshot(type='png')
                if await asyncio.to_thread(classify_binary, model, image):
                    await human_click(self.page, await tile.bounding_box())
        else:
            canvas = images.first
            box = await canvas.bounding_box()
            if not box: raise SolveFailure('Challenge canvas is not visible')
            img = await canvas.screenshot(type='png', scale='css')
            points = await asyncio.to_thread(detect_yolo, model, img, subject(prompt))
            if not points: raise SolveFailure('No matching YOLO detections')
            for point in points: await human_click(self.page, box, point)
        button = frame.locator('.button-submit')
        await human_click(self.page, await button.bounding_box())

class HCaptchaSolver:
    def __init__(self, models: Path, headless=True, proxy=None, driver_factory=BrowserChallenge):
        self.registry = Registry(models); self.headless = headless; self.proxy = proxy
        self.driver_factory = driver_factory
    async def solve(self, url: str, sitekey: str | None = None) -> str:
        if sitekey:
            log.info('sitekey is not injected; target page must already contain its widget')
        proxy = None
        if self.proxy:
            p = urlsplit(self.proxy)
            if p.scheme not in ('http', 'https', 'socks5') or not p.hostname or not p.port:
                raise ValueError('proxy must be an http, https, or socks5 URL with host and port')
            proxy = {'server': f'{p.scheme}://{p.hostname}:{p.port}'}
            if p.username: proxy['username'] = p.username
            if p.password: proxy['password'] = p.password
        last = None
        for attempt in range(3):
            try:
                async with async_playwright() as playwright:
                    browser = await playwright.firefox.launch(headless=self.headless, proxy=proxy)
                    try:
                        page = await browser.new_page()
                        driver = self.driver_factory(page)
                        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                        await driver.start()
                        for _ in range(5):
                            token = await driver.token()
                            if token: return token
                            await driver.run_round(self.registry)
                            for _ in range(10):
                                await asyncio.sleep(0.5)
                                token = await driver.token()
                                if token: return token
                        raise SolveFailure('Challenge did not produce a token')
                    finally: await browser.close()
            except (ModelUnavailable, UnsupportedChallenge, ValueError): raise
            except (PlaywrightTimeout, SolveFailure) as exc:
                last = exc; log.warning('solve attempt %d failed: %s', attempt + 1, exc)
                if attempt < 2: await asyncio.sleep(2 ** attempt)
        raise SolveFailure(f'Failed after 3 attempts: {last}') from last
ARENA_FILE_END
RUN cat <<'ARENA_FILE_END' > /app/registry_builder.py
import re
from pathlib import Path

def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()

def build_registry(names):
    """Conservative filename aliases, not a substitute for upstream semantic metadata."""
    entries = []
    for name in sorted(names):
        if Path(name).name != name or not name.lower().endswith('.onnx'): continue
        stem = name[:-5].lower()
        if any(x in stem for x in ('clip', '-seg', '_seg', 'yolov5', 'yolov6', 'yolov7')): continue
        is_yolo = bool(re.search(r'_yolov8[nsmlx](?:\b|$)', stem))
        if 'yolo' in stem and not is_yolo: continue
        label = re.sub(r'_yolov8[nsmlx].*$', '', stem) if is_yolo else stem
        label = re.sub(r'\d{4,}$', '', label)
        label = normalize(label)
        if label:
            entries.append({'kind': 'area_select' if is_yolo else 'binary', 'label': label, 'model': name})
    return {'version': 1, 'entries': entries}
ARENA_FILE_END
RUN cat <<'ARENA_FILE_END' > /app/vendor/archived_onnx/resnet.py
# -*- coding: utf-8 -*-
# Time       : 2022/4/30 17:00
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import cv2
import numpy as np
from cv2.dnn import Net
from loguru import logger
from scipy.special import expit


class ChallengeStyle:
    WATERMARK = 144  # onTrigger 128x128
    GENERAL = 128
    GAN = 144


@dataclass
class ResNetControl:
    net: Net

    @classmethod
    def from_pluggable_model(cls, net: Net):
        return cls(net=net)

    def binary_classify(self, img_stream: Any) -> bool | Tuple[bool, np.ndarray]:
        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)

        if img.shape[0] == ChallengeStyle.WATERMARK:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        img = cv2.resize(img, (64, 64))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 64), (0, 0, 0), swapRB=True, crop=False)

        # Use the delayed reflection mechanism
        # to ensure the consistency of both ends of the distributed network
        if self.net is None:
            logger.debug("The remote network does not exist or the local cache has expired.")
            return False
        self.net.setInput(blob)
        out = self.net.forward()
        logit = out[0]
        proba = expit(logit)
        if not np.argmax(out, axis=1)[0]:
            return True, proba
        return False, proba

    def execute(self, img_stream: bytes, **kwargs) -> bool | Tuple[bool, np.ndarray]:
        """Implementation process of solution"""
        get_proba = kwargs.get("proba", False)
        result = self.binary_classify(img_stream)
        if result is None or isinstance(result, bool):
            return result
        if isinstance(result, tuple):
            prediction, confidence = result
            if get_proba is True:
                return prediction, confidence
            return prediction
ARENA_FILE_END
RUN cat <<'ARENA_FILE_END' > /app/vendor/archived_onnx/yolo.py
# -*- coding: utf-8 -*-
# Time       : 2022/3/2 0:52
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from typing import Literal

import cv2
import numpy as np
from onnxruntime import InferenceSession

from .utils import nms, xywh2xyxy, draw_detections, sigmoid, multiclass_nms


@dataclass
class YOLOv8:
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    classes: List[str] = field(default_factory=list)

    session: InferenceSession = None

    input_names = None
    input_shape = None
    input_height = None
    input_width = None
    output_names = None

    img_height = None
    img_width = None

    def __post_init__(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @classmethod
    def from_pluggable_model(cls, session: InferenceSession, classes: List[str]):
        return cls(session=session, classes=classes)

    def __call__(self, image: Path | bytes, shape_type: Literal["point", "bounding_box"] = "point"):
        if isinstance(image, Path):
            image = image.read_bytes()

        np_array = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(np_array, flags=1)

        boxes, scores, class_ids = self.detect_objects(image)

        response = []
        if shape_type == "point":
            for i, class_id in enumerate(class_ids):
                x1, y1, x2, y2 = boxes[i]
                center_x, center_y = int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                response.append((self.classes[class_id], (center_x, center_y), scores[i]))
        elif shape_type == "bounding_box":
            for i, class_id in enumerate(class_ids):
                x1, y1, x2, y2 = boxes[i]
                point_start, point_end = (x1, y1), (x2, y2)
                response.append((self.classes[class_id], point_start, point_end, scores[i]))
        return response

    def detect_objects(self, image: np.ndarray):
        self.img_height, self.img_width = image.shape[:2]

        input_tensor = self._prepare_input(image)

        # Perform inference on the image
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        boxes, scores, class_ids = self._process_output(outputs)

        return boxes, scores, class_ids

    def _prepare_input(self, image):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def _process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]


def is_matched_ash_of_war(ash: str, class_name: str):
    if "head of " in ash:
        if "head" not in class_name:
            return False
        keyword = class_name.replace("-head", "").strip()
        if keyword not in ash:
            return False
        return True

    # catch-all rules
    if class_name not in ash:
        return False
    return True


def finetune_keypoint(name: str, point: List[int, int]) -> List[int, int]:
    point = point.copy()
    if name in ["nine", "9"]:
        point[-1] = point[-1] + 8
        point[0] = point[0] + 2
    if name in ["two", "2"]:
        point[-1] = point[-1] + 7
        point[0] = point[0] + 4
    return point


@dataclass
class YOLOv8Seg:
    conf_threshold: float = 0.71
    iou_threshold: float = 0.5
    num_masks: int = 32
    classes: List[str] = field(default_factory=list)
    session: InferenceSession = None

    input_names = None
    input_shape = None
    input_height = None
    input_width = None
    output_names = None

    img_height = None
    img_width = None
    boxes = None
    scores = None
    class_ids = None
    mask_maps = None

    colors = None

    def __post_init__(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        if not self.colors:
            rng = np.random.default_rng(3)
            self.colors = rng.uniform(0, 255, size=(len(self.classes), 3))

    @classmethod
    def from_pluggable_model(cls, session: InferenceSession, classes: List[str]):
        return cls(session=session, classes=classes)

    def __call__(
        self, image: Path | bytes, shape_type: Literal["point", "bounding_box"] = "point", **kwargs
    ):
        if isinstance(image, Path):
            image = image.read_bytes()

        np_array = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(np_array, flags=1)

        boxes, scores, class_ids, mask_maps = self.segment_objects(image)

        response = []
        if shape_type == "point":
            for i, class_id in enumerate(class_ids):
                x1, y1, x2, y2 = boxes[i]
                center_x, center_y = int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1)
                response.append((self.classes[class_id], (center_x, center_y), scores[i]))
        elif shape_type == "bounding_box":
            for i, class_id in enumerate(class_ids):
                x1, y1, x2, y2 = boxes[i]
                point_start, point_end = (x1, y1), (x2, y2)
                response.append((self.classes[class_id], point_start, point_end, scores[i]))
        return response

    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_box_output(self, box_output):
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(
            self.boxes, (self.img_height, self.img_width), (mask_height, mask_width)
        )

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(
                scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC
            )

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(
            boxes, (self.input_height, self.input_width), (self.img_height, self.img_width)
        )

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, mask_alpha=0.4):
        return draw_detections(
            image, self.boxes, self.scores, self.class_ids, self.colors, self.classes, mask_alpha
        )

    def draw_masks(self, image, mask_alpha=0.5):
        return draw_detections(
            image,
            self.boxes,
            self.scores,
            self.class_ids,
            self.colors,
            self.classes,
            mask_alpha,
            mask_maps=self.mask_maps,
        )

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes
ARENA_FILE_END
RUN cat <<'ARENA_FILE_END' > /app/vendor/archived_onnx/utils.py
# -*- coding: utf-8 -*-
# Time       : 2023/10/15 21:37
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import os

import cv2
import importlib_metadata
import numpy as np

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


def _is_package_available(pkg_name: str) -> bool:
    try:
        package_version = importlib_metadata.metadata(pkg_name) is not None
        return bool(package_version)
    except importlib_metadata.PackageNotFoundError:
        return False


_torch_available = False
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = _is_package_available("torch")
else:
    _torch_available = False

_transformers_available = _is_package_available("transformers")


def is_torch_available():
    return _torch_available


def is_transformers_available():
    return _transformers_available


is_cuda_pipline_available = is_torch_available() and is_transformers_available()


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_detections(
    image, boxes, scores, class_ids, colors, classes, mask_alpha=0.3, mask_maps=None
):
    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, class_ids, colors, mask_alpha, mask_maps)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = classes[class_id]
        caption = f"{label} {int(score * 100)}%"
        (tw, th), _ = cv2.getTextSize(
            text=caption,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            thickness=text_thickness,
        )
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(
            mask_img,
            caption,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return mask_img


def draw_masks(image, boxes, class_ids, colors, mask_alpha=0.3, mask_maps=None):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(
        text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontsize, thickness=text_thickness
    )
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(
        img1,
        (x1 - offset * 2, y1 + offset),
        (x1 + tw + offset * 2, y1 - th - offset),
        (0, 115, 255),
        -1,
    )
    cv2.putText(
        img1, name1, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, fontsize, (255, 255, 255), text_thickness
    )

    (tw, th), _ = cv2.getTextSize(
        text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontsize, thickness=text_thickness
    )
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(
        img2,
        (x1 - offset * 2, y1 + offset),
        (x1 + tw + offset * 2, y1 - th - offset),
        (94, 23, 235),
        -1,
    )

    cv2.putText(
        img2, name2, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, fontsize, (255, 255, 255), text_thickness
    )

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img
ARENA_FILE_END
RUN cat <<'ARENA_FILE_END' > /app/LICENSE
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<https://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<https://www.gnu.org/licenses/why-not-lgpl.html>.
ARENA_FILE_END
RUN cat <<'ARENA_FILE_END' > /app/install_models.py
"""Build-time download and CPU validation of every ONNX release asset."""
import gc
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from registry_builder import build_registry

API = 'https://api.github.com/repos/QIN2DIM/hcaptcha-challenger/releases/tags/model'
DEST = Path('/app/models')
DTYPES = {
    'tensor(float)': np.float32, 'tensor(double)': np.float64,
    'tensor(float16)': np.float16, 'tensor(int64)': np.int64,
    'tensor(int32)': np.int32, 'tensor(int16)': np.int16,
    'tensor(int8)': np.int8, 'tensor(uint8)': np.uint8,
    'tensor(uint16)': np.uint16, 'tensor(uint32)': np.uint32,
    'tensor(uint64)': np.uint64, 'tensor(bool)': np.bool_,
    'tensor(string)': np.object_,
}

def dummy_input(item):
    if item.type not in DTYPES:
        raise ValueError(f'Unsupported ONNX input type {item.type} for {item.name}')
    # Dynamic dimensions require concrete values. This generic choice is not
    # sufficient for every model; incompatible models are skipped.
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in item.shape]
    if item.type == 'tensor(string)':
        return np.full(shape, '', dtype=np.object_)
    return np.zeros(shape, dtype=DTYPES[item.type])

def install():
    DEST.mkdir(parents=True, exist_ok=True)
    http = requests.Session()
    http.headers.update({'User-Agent': 'onnx-model-build', 'Accept': 'application/vnd.github+json'})
    try:
        response = http.get(API, timeout=(30, 120))
        response.raise_for_status()
        release = response.json()
        assets = [a for a in release['assets'] if a['name'].lower().endswith('.onnx')]
    except Exception as exc:
        raise RuntimeError(f'Cannot retrieve model release from {API}: {exc}') from exc
    names = set()
    validated = []
    skipped = 0
    for index, asset in enumerate(assets, 1):
        name = asset.get('name', '<missing>')
        temp = None
        session = feed = result = None
        try:
            url = asset['browser_download_url']
            if Path(name).name != name or name in names:
                raise ValueError(f'Invalid or duplicate asset name: {name!r}')
            names.add(name)
            path = DEST / name
            temp = DEST / (name + '.part')
            print(f'Downloading {index}/{len(assets)}: {url}', flush=True)
            with http.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with temp.open('wb') as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if temp.stat().st_size <= 0:
                raise ValueError('empty file')
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            session = ort.InferenceSession(str(temp), sess_options=opts,
                                           providers=['CPUExecutionProvider'])
            inputs = session.get_inputs()
            if not inputs or not session.get_outputs():
                raise ValueError('model has no inputs or outputs')
            feed = {item.name: dummy_input(item) for item in inputs}
            result = session.run(None, feed)
            print(f'Validated {name}: {temp.stat().st_size} bytes; '
                  f'inputs={[(k, v.shape, str(v.dtype)) for k, v in feed.items()]}; '
                  f'output shapes={[getattr(x, "shape", None) for x in result]}', flush=True)
            session = feed = result = None
            gc.collect()
            os.replace(temp, path)
            validated.append(name)
        except Exception as exc:
            skipped += 1
            print(f'Warning: Skipping model {name} due to download or validation error: {exc}', flush=True)
        finally:
            session = feed = result = None
            gc.collect()
            if temp is not None:
                temp.unlink(missing_ok=True)
    print(f'Successfully validated {len(validated)} models. Skipped {skipped} models.', flush=True)
    if not validated:
        raise RuntimeError(f'No ONNX models were successfully validated from {API}')
    (DEST / 'registry.json').write_text(json.dumps(build_registry(validated), indent=2) + '\n')

if __name__ == '__main__':
    install()
ARENA_FILE_END
RUN python /app/install_models.py && rm /app/install_models.py
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
