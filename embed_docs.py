#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 커밋확인3
"""
embed_docs.py
- 기본 경로/컬렉션명 유지
- 속도 상한(34p 기준 1~3분대)을 우선한 실전형
- body_text + light OCR + visual_summary + retrieval_text 생성
- PaddleOCR 우선, 실패 시 Tesseract fallback
- PaddleOCR 버전별 인자 차이를 피하기 위해 동적 introspection 기반 초기화
- 전체 페이지는 가볍게 1회 OCR
- ERP/UI 스크린샷이 있는 페이지만 제한적으로 정밀 OCR
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import os
import inspect
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
from sentence_transformers import SentenceTransformer

try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
except Exception:
    torch = None

try:
    import paddle
except Exception:
    paddle = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None


LOG = logging.getLogger("embed_docs")

DEFAULT_INPUT_DIR = Path("/home/siwasoft/gsllm/embed_test/data/IT_file")
DEFAULT_WORK_DIR = Path("/home/siwasoft/gsllm/embed_test/work/collection_A")
DEFAULT_CHROMA_DIR = Path("/home/siwasoft/gsllm/embed_test/chroma/collection_A")
DEFAULT_COLLECTION = "collection_A"
DEFAULT_EMBED_MODEL = "/home/siwasoft/bge-m3"

OCR_ENGINE = None
OCR_ENGINE_NAME = "tesseract"
OCR_DEVICE = "cpu"


@dataclass
class Section:
    source_type: str
    text: str
    title: str
    page_num: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class ExtractedDocument:
    file_path: Path
    file_type: str
    body_sections: List[Section]
    ocr_sections: List[Section]
    visual_sections: List[Section]
    retrieval_sections: List[Section]
    meta: Dict[str, Any]
    # 업로드 원본 파일명(청크 metadata.original_filename); 미지정 시 file_path.name 과 동일
    original_filename: str


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_ocr_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    replacements = {
        "조회클릭": "조회 클릭",
        "신규클릭": "신규 클릭",
        "수정클릭": "수정 클릭",
        "저장클릭": "저장 클릭",
        "내역조회클릭": "내역조회 클릭",
        "메뉴위치": "메뉴 위치",
        "입고업체관리": "입고업체 관리",
        "판매업체관리": "판매업체 관리",
        "물류센터관리": "물류센터 관리",
        "사용여부": "사용 여부",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    regex_replacements = [
        (r"P\s*O\s*S|P0S|PO5", "POS"),
        (r"S\s*C\s*M", "SCM"),
        (r"사\s*용\s*여\s*부", "사용 여부"),
        (r"업\s*체\s*명", "업체명"),
        (r"메\s*뉴\s*위\s*치", "메뉴 위치"),
        (r"운\s*송\s*장\s*배\s*정", "운송장 배정"),
    ]
    for pat, rep in regex_replacements:
        text = re.sub(pat, rep, text, flags=re.I)
    text = re.sub(r"\s*→\s*", " → ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def clean_for_embedding(text: str) -> str:
    text = normalize_ocr_text(text)
    if not text:
        return ""
    text = re.sub(r"([가-힣A-Za-z0-9])([:()\[\]/\-])", r"\1 \2", text)
    text = re.sub(r"([:()\[\]/\-])([가-힣A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([가-힣])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([가-힣])", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def is_meaningful_text(text: str, min_len: int = 8) -> bool:
    text = clean_for_embedding(text)
    if len(text) < min_len:
        return False
    useful = len(re.findall(r"[가-힣A-Za-z0-9]", text))
    return useful >= max(4, int(len(text) * 0.18))


def split_text(text: str, max_chars: int = 1400, overlap: int = 180) -> List[str]:
    text = clean_for_embedding(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text]

    chunks: List[str] = []
    buf = ""
    for para in paras:
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + "\n\n" + para).strip() if tail else para
            if len(buf) <= max_chars:
                continue
        start = 0
        while start < len(para):
            end = min(start + max_chars, len(para))
            piece = para[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(para):
                buf = ""
                break
            start = max(0, end - overlap)
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def ocr_quality_score(text: str) -> float:
    s = clean_for_embedding(text)
    if not s:
        return 0.0
    total = len(s)
    useful = len(re.findall(r"[가-힣A-Za-z0-9]", s))
    hangul = len(re.findall(r"[가-힣]", s))
    weird = len(re.findall(r"[^가-힣A-Za-z0-9\s\-_/,.:()%\[\]→•·+]", s))
    score = 0.0
    score += min(1.0, useful / max(1, total)) * 0.50
    score += min(1.0, hangul / max(1, total * 0.18)) * 0.20
    score += max(0.0, 1.0 - weird / max(1, total)) * 0.30
    return round(score, 4)


def keep_ocr_text(text: str, *, min_len: int = 4, min_score: float = 0.08) -> bool:
    s = clean_for_embedding(text)
    if len(s) < min_len:
        return False
    if re.search(r"(조회|신규|수정|저장|입력|선택|확인|POS|SCM|코드|관리|필수|운송장|피킹|배송|업체명|사용 여부)", s, flags=re.I):
        return True
    return ocr_quality_score(s) >= min_score


def fuzzy_sim(a: str, b: str) -> float:
    a = re.sub(r"\s+", "", clean_for_embedding(a or ""))
    b = re.sub(r"\s+", "", clean_for_embedding(b or ""))
    if not a or not b:
        return 0.0
    common = sum(1 for x, y in zip(a, b) if x == y)
    return common / max(len(a), len(b))


def dedup_similar_lines(lines: Sequence[str], threshold: float = 0.92) -> List[str]:
    out: List[str] = []
    for line in lines:
        s = clean_for_embedding(line)
        if not s:
            continue
        if any(fuzzy_sim(s, prev) >= threshold for prev in out):
            continue
        out.append(s)
    return out


def infer_menu_path(text: str) -> str:
    text = clean_for_embedding(text)
    patterns = [
        r"(설정\s*[–-]\s*[가-힣A-Za-z0-9 ]+\s*[–-]\s*[가-힣A-Za-z0-9 ()/]+\s*(?:[–-]\s*[가-힣A-Za-z0-9 ()/]+)?)",
        r"(물류\s*관리\s*[–-]\s*[가-힣A-Za-z0-9 ]+\s*[–-]\s*[가-힣A-Za-z0-9 ()/]+\s*(?:[–-]\s*[가-힣A-Za-z0-9 ()/]+)?)",
        r"(발주\s*관리\s*[–-]\s*[가-힣A-Za-z0-9 ]+\s*[–-]\s*[가-힣A-Za-z0-9 ()/]+\s*(?:[–-]\s*[가-힣A-Za-z0-9 ()/]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return re.sub(r"\s*[–-]\s*", " > ", m.group(1)).strip()
    return ""


def extract_step_numbers(text: str) -> List[str]:
    text = clean_for_embedding(text)
    found = re.findall(r"(?:^|\s)([1-9][0-9]?)\s*(?:[.)]|)(?=\s*(?:조회|신규|수정|저장|클릭|입력|선택|확인))", text)
    out: List[str] = []
    for x in found:
        if x not in out:
            out.append(x)
    return out[:6]


def extract_actions(text: str) -> List[str]:
    s = clean_for_embedding(text)
    out: List[str] = []
    for kw in ["조회", "신규", "수정", "저장", "입력", "선택", "확인", "스캔", "출력"]:
        if kw in s and kw not in out:
            out.append(kw)
    return out[:8]


def extract_field_keywords(text: str) -> List[str]:
    s = clean_for_embedding(text)
    out: List[str] = []
    for kw in ["업체명", "사용 여부", "코드", "물류센터", "브랜드", "POS", "SCM", "운송장", "택배사", "주문번호", "차수", "피킹", "배송", "재고", "입고", "판매업체", "입고업체"]:
        if kw in s and kw not in out:
            out.append(kw)
    return out[:8]


def derive_screen_title(body_text: str, menu_path: str, ocr_text: str = "") -> str:
    body = clean_for_embedding(body_text)
    ocr = clean_for_embedding(ocr_text)
    menu_like = re.findall(r"([가-힣A-Za-z0-9 ]+[–-][가-힣A-Za-z0-9 ]+[–-][가-힣A-Za-z0-9 ()/]+(?:[–-][가-힣A-Za-z0-9 ()/]+)?)", body)
    if menu_like:
        last = re.split(r"\s*[–-]\s*", menu_like[0])[-1].strip()
        if last:
            return last
    if menu_path:
        nodes = [x.strip() for x in menu_path.split(">") if x.strip()]
        if nodes:
            return nodes[-1]
    m = re.search(r"([가-힣A-Za-z0-9 ]+(?:관리|현황|리스트|확정|등록|배정)(?:\s*\(\s*[^)]+\s*\))?)", ocr)
    return m.group(1).strip() if m else ""


def render_pdf_page(page: Any, dpi: int = 140) -> Optional[Image.Image]:
    try:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None


def render_pdf_page_hires(page: Any, dpi: int = 210) -> Optional[Image.Image]:
    try:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None


def enhance_gray(img: Image.Image, scale: float = 2.0, threshold: bool = True) -> Image.Image:
    w, h = img.size
    out = img.convert("L")
    out = out.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    out = ImageEnhance.Contrast(out).enhance(1.8)
    out = ImageEnhance.Sharpness(out).enhance(1.8)
    if cv2 is not None and threshold:
        arr = np.array(out)
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
        arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        out = Image.fromarray(arr)
    return out


def deskew_image(img: Image.Image) -> Image.Image:
    if cv2 is None:
        return img
    gray = np.array(img.convert("L"))
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3:
        return img
    arr = np.array(img)
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)


def _try_set_paddle_device(use_gpu: bool) -> str:
    if paddle is None:
        return "cpu"
    try:
        if use_gpu and paddle.device.is_compiled_with_cuda():
            paddle.set_device("gpu:0")
            return "gpu"
        paddle.set_device("cpu")
        return "cpu"
    except Exception:
        return "cpu"


def init_ocr_engine(use_gpu: bool = True) -> Tuple[str, str]:
    global OCR_ENGINE, OCR_ENGINE_NAME, OCR_DEVICE
    OCR_ENGINE = None
    OCR_ENGINE_NAME = "tesseract"
    OCR_DEVICE = "cpu"

    if PaddleOCR is not None and paddle is not None:
        try:
            device = _try_set_paddle_device(use_gpu)
            sig = inspect.signature(PaddleOCR.__init__)
            allowed = set(sig.parameters.keys())
            kwargs: Dict[str, Any] = {}

            if "lang" in allowed:
                kwargs["lang"] = "korean"
            elif "ocr_version" in allowed:
                kwargs["ocr_version"] = "PP-OCRv4"

            if "use_textline_orientation" in allowed:
                kwargs["use_textline_orientation"] = False
            elif "use_angle_cls" in allowed:
                kwargs["use_angle_cls"] = False

            OCR_ENGINE = PaddleOCR(**kwargs)
            OCR_ENGINE_NAME = "paddleocr"
            OCR_DEVICE = device
            return OCR_ENGINE_NAME, OCR_DEVICE
        except Exception as e:
            LOG.warning("PaddleOCR init failed, fallback to Tesseract: %s", e)

    OCR_ENGINE = None
    OCR_ENGINE_NAME = "tesseract"
    OCR_DEVICE = "cpu"
    return OCR_ENGINE_NAME, OCR_DEVICE


def run_tesseract_ocr(img: Image.Image, lang: str = "kor+eng", psm: int = 6) -> str:
    try:
        text = pytesseract.image_to_string(img, lang=lang, config=f"--oem 3 --psm {psm}")
    except Exception:
        return ""
    return normalize_ocr_text(text)


def run_paddle_ocr(img: Image.Image) -> str:
    global OCR_ENGINE
    if OCR_ENGINE is None:
        return ""
    try:
        arr = np.array(img.convert("RGB"))
        result = OCR_ENGINE.ocr(arr)
        if not result or not result[0]:
            return ""
        texts: List[str] = []
        for item in result[0]:
            if item and len(item) >= 2 and item[1]:
                texts.append(item[1][0])
        return normalize_ocr_text("\n".join(texts))
    except Exception:
        return ""


def run_ocr_fast(img: Image.Image, *, lang: str = "kor+eng", psm: int = 6, hires: bool = False) -> str:
    variant = enhance_gray(img, scale=2.6 if hires else 2.0, threshold=True)
    paddle_text = run_paddle_ocr(variant) if OCR_ENGINE_NAME == "paddleocr" else ""
    tess_text = "" if OCR_ENGINE_NAME == "paddleocr" else run_tesseract_ocr(variant, lang=lang, psm=psm)
    candidates = [x for x in [paddle_text, tess_text] if keep_ocr_text(x, min_len=3, min_score=0.05)]
    if not candidates:
        return clean_for_embedding(paddle_text or tess_text or "")
    return max(candidates, key=ocr_quality_score)


def detect_red_regions(img: Image.Image) -> Tuple[int, float, List[Tuple[int, int, int, int]]]:
    if cv2 is None:
        return 0, 0.0, []
    rgb = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 28, 35], dtype=np.uint8)
    upper1 = np.array([18, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 28, 35], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    H, W = mask.shape[:2]
    red_ratio = float(np.count_nonzero(mask)) / max(1.0, float(H * W))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    min_area = max(80, int(W * H * 0.00008))
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area and w >= 8 and h >= 6:
            boxes.append((x, y, x + w, y + h))
    boxes = sorted(boxes, key=lambda b: ((b[1] + b[3]) // 2, (b[0] + b[2]) // 2))
    return len(boxes[:2]), round(red_ratio, 6), boxes[:2]


def crop_with_padding(img: Image.Image, box: Tuple[int, int, int, int], pad: int = 24) -> Image.Image:
    x1, y1, x2, y2 = box
    return img.crop((max(0, x1 - pad), max(0, y1 - pad), min(img.width, x2 + pad), min(img.height, y2 + pad)))


def split_vertical_regions(img: Image.Image) -> List[Tuple[str, Image.Image, int]]:
    w, h = img.size
    return [("top_region", img.crop((0, 0, w, int(h * 0.25))), 6)]


def detect_erp_screenshot_region(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    if cv2 is None:
        return None
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1.0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < W * H * 0.10 or area > W * H * 0.82 or y < int(H * 0.10):
            continue
        aspect = w / max(1, h)
        if aspect < 1.0 or aspect > 5.0:
            continue
        roi = gray[y:y+h, x:x+w]
        edge_density = np.count_nonzero(cv2.Canny(roi, 50, 150)) / max(1, roi.size)
        score = (area / (W * H)) * 0.6 + edge_density * 2.2
        if score > best_score:
            best_score = score
            best_box = (x, y, x + w, y + h)
    if best_box is None:
        return None
    x1, y1, x2, y2 = best_box
    pad_x = int((x2 - x1) * 0.02)
    pad_y = int((y2 - y1) * 0.02)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y), min(img.width, x2 + pad_x), min(img.height, y2 + pad_y))


def split_erp_regions(erp_img: Image.Image) -> List[Tuple[str, Image.Image, int]]:
    w, h = erp_img.size
    return [
        ("erp_toolbar", erp_img.crop((0, 0, w, int(h * 0.16))), 7),
        ("erp_left_grid", erp_img.crop((0, int(h * 0.14), int(w * 0.34), h)), 6),
        ("erp_right_form", erp_img.crop((int(w * 0.31), int(h * 0.14), w, h)), 6),
    ]


def classify_page_kind(body_text: str, ocr_text: str, red_regions: int) -> str:
    combined = clean_for_embedding("\n".join([body_text or "", ocr_text or ""]))
    if re.search(r"(조회|신규|수정|저장)", combined) and re.search(r"(관리|업체|창고|브랜드|사용자|권한|운송장|배송|피킹)", combined):
        return "manual_ui"
    if red_regions >= 1 and re.search(r"(조회|신규|수정|저장|클릭|입력|선택)", combined):
        return "manual_ui"
    if re.search(r"(메뉴 위치|조회 클릭|신규 클릭|수정 클릭|저장 클릭)", combined):
        return "manual_ui"
    if re.search(r"(contents|보고서|현황|수불추적|재고)", combined, flags=re.I):
        return "report_or_table"
    return "general"


def should_run_targeted_ocr(body_text: str, erp_box: Optional[Tuple[int, int, int, int]], red_regions: int) -> bool:
    body = clean_for_embedding(body_text)
    return bool(erp_box is not None or red_regions > 0 or re.search(r"(조회|신규|저장|메뉴 위치|POS|SCM|운송장|피킹|배송)", body))


def build_visual_summary(*, body_text: str, ocr_text: str, red_regions: int, crop_ocr_texts: Sequence[str], page_kind: str, erp_detected: bool = False, erp_text: str = "") -> str:
    body_text = clean_for_embedding(body_text)
    ocr_text = clean_for_embedding(ocr_text)
    erp_text = clean_for_embedding(erp_text)
    combined = "\n".join([body_text, ocr_text, erp_text] + [clean_for_embedding(x) for x in crop_ocr_texts if x])
    menu_path = infer_menu_path(combined)
    step_numbers = extract_step_numbers(combined)
    actions = extract_actions(combined)
    fields = extract_field_keywords(combined)
    bits: List[str] = []
    if menu_path:
        bits.append(f"메뉴 경로는 {menu_path} 입니다.")
    if page_kind == "manual_ui":
        bits.append("이 페이지는 화면 사용법을 설명하는 매뉴얼형 UI 페이지로 보입니다.")
    if step_numbers:
        bits.append(f"페이지에는 단계 번호 {', '.join(step_numbers)} 가 나타납니다.")
    if erp_detected:
        bits.append("페이지 내 ERP 시스템 화면 캡처가 포함되어 있습니다.")
    if red_regions > 0:
        bits.append(f"빨간 강조 영역이 {red_regions} 개 감지되었습니다.")
    if actions:
        bits.append(f"화면에서 {', '.join(actions[:6])} 기능이 확인됩니다.")
    if fields:
        bits.append(f"주요 필드 또는 항목으로 {', '.join(fields[:6])} 이 확인됩니다.")
    useful_crops = [x for x in crop_ocr_texts if keep_ocr_text(x, min_len=8, min_score=0.12)]
    if useful_crops:
        bits.append(f"강조 영역 OCR 요약 : {' | '.join(useful_crops[:1])[:140]}")
    if not bits:
        bits.append("이 페이지는 문서 본문 중심 페이지입니다.")
    return clean_for_embedding(" ".join(bits))


def build_retrieval_text(*, page_num: int, body_text: str, ocr_text: str, visual_text: str, menu_path: str, page_kind: str, screen_title: str, erp_text: str, red_regions: int) -> str:
    body = clean_for_embedding(body_text)
    visual = clean_for_embedding(visual_text)
    erp = clean_for_embedding(erp_text)
    ocr = clean_for_embedding(ocr_text)
    actions = extract_actions("\n".join([body, ocr, erp]))
    fields = extract_field_keywords("\n".join([body, ocr, erp]))
    core_lines: List[str] = []
    for ln in body.splitlines():
        s = clean_for_embedding(ln)
        if s and re.search(r"(메뉴 위치|조회 클릭|신규 클릭|수정 클릭|저장 클릭|필수 입력|POS|SCM|관리할 수 있는 메뉴|현황|출력|스캔)", s):
            core_lines.append(s)
    core_lines = dedup_similar_lines(core_lines, threshold=0.92)[:4]
    erp_lines: List[str] = []
    for ln in erp.splitlines():
        s = clean_for_embedding(ln)
        if s and re.search(r"(조회|신규|수정|저장|업체명|사용 여부|코드|POS|SCM|운송장|차수|물류센터)", s):
            erp_lines.append(s)
    erp_lines = dedup_similar_lines(erp_lines, threshold=0.92)[:2]

    parts = [f"페이지 {page_num}."]
    if menu_path:
        parts.append(f"메뉴 경로 {menu_path}.")
    if page_kind:
        parts.append(f"페이지 유형 {page_kind}.")
    if screen_title:
        parts.append(f"화면명 {screen_title}.")
    if actions:
        parts.append(f"주요 동작 {', '.join(actions[:6])}.")
    if fields:
        parts.append(f"주요 필드 {', '.join(fields[:6])}.")
    if red_regions > 0:
        parts.append(f"강조 영역 {red_regions} 개.")
    if visual:
        parts.append(visual)
    if core_lines:
        parts.append("핵심 텍스트 " + " | ".join(core_lines) + ".")
    if erp_lines:
        parts.append("화면 내부 OCR " + " | ".join(erp_lines) + ".")
    return clean_for_embedding(" ".join(parts))


def extract_erp_ocr_texts(img: Image.Image, *, ocr_lang: str, erp_box: Tuple[int, int, int, int]) -> Dict[str, Any]:
    out = {"erp_detected": True, "erp_box": erp_box, "erp_text": ""}
    erp_crop = deskew_image(crop_with_padding(img, erp_box, pad=8))
    texts: List[str] = []

    full_text = run_ocr_fast(erp_crop, lang=ocr_lang, psm=6, hires=True)
    if keep_ocr_text(full_text, min_len=5, min_score=0.10):
        texts.append(full_text)

    for _, region_img, psm in split_erp_regions(erp_crop):
        txt = run_ocr_fast(region_img, lang=ocr_lang, psm=psm, hires=True)
        if keep_ocr_text(txt, min_len=5, min_score=0.10):
            texts.append(txt)

    merged_lines: List[str] = []
    for txt in texts:
        merged_lines.extend([x.strip() for x in txt.splitlines() if x.strip()])
    merged_lines = dedup_similar_lines(merged_lines, threshold=0.92)
    out["erp_text"] = clean_for_embedding("\n".join(merged_lines[:60]))
    return out


def extract_pdf_document(
    file_path: Path,
    enable_ocr: bool,
    ocr_lang: str,
    save_page_images_dir: Optional[Path] = None,
    original_filename: Optional[str] = None,
) -> ExtractedDocument:
    body_sections: List[Section] = []
    ocr_sections: List[Section] = []
    visual_sections: List[Section] = []
    retrieval_sections: List[Section] = []

    resolved_original = (original_filename or "").strip() or file_path.name

    meta = {"pages": 0, "renderer": "fitz", "ocr_engine": OCR_ENGINE_NAME}
    doc = fitz.open(str(file_path))
    meta["pages"] = len(doc)

    for i, page in enumerate(doc, start=1):
        print(f"  [PAGE] {i}/{len(doc)}")
        body_text = clean_for_embedding(page.get_text("text") or "")
        img = render_pdf_page(page)

        if save_page_images_dir and img is not None:
            ensure_dir(save_page_images_dir)
            img.save(save_page_images_dir / f"page_{i:04d}.png")

        ocr_text = ""
        crop_ocr_texts: List[str] = []
        red_regions = 0
        red_ratio = 0.0
        erp_detected = False
        erp_text = ""

        if enable_ocr and img is not None:
            base_img = deskew_image(img)
            base_ocr = run_ocr_fast(base_img, lang=ocr_lang, psm=6, hires=False)
            if keep_ocr_text(base_ocr, min_len=4, min_score=0.06):
                ocr_text = base_ocr

            red_regions, red_ratio, red_boxes = detect_red_regions(base_img)
            erp_box = detect_erp_screenshot_region(base_img)
            targeted = should_run_targeted_ocr(body_text, erp_box, red_regions)

            region_texts: List[str] = []
            if targeted:
                top_img = split_vertical_regions(base_img)[0][1]
                top_ocr = run_ocr_fast(top_img, lang=ocr_lang, psm=6, hires=False)
                if keep_ocr_text(top_ocr, min_len=4, min_score=0.08):
                    region_texts.append(top_ocr)

            if erp_box is not None:
                erp_info = extract_erp_ocr_texts(base_img, ocr_lang=ocr_lang, erp_box=erp_box)
                erp_detected = True
                erp_text = clean_for_embedding(erp_info.get("erp_text", "") or "")

            for box in red_boxes[:2]:
                txt = run_ocr_fast(crop_with_padding(base_img, box, pad=28), lang=ocr_lang, psm=6, hires=True)
                if keep_ocr_text(txt, min_len=4, min_score=0.10):
                    crop_ocr_texts.append(txt)

            if not is_meaningful_text(body_text, min_len=70) and targeted:
                hires = render_pdf_page_hires(page)
                if hires is not None:
                    extra_page_ocr = run_ocr_fast(deskew_image(hires), lang=ocr_lang, psm=6, hires=False)
                    if keep_ocr_text(extra_page_ocr, min_len=5, min_score=0.08):
                        region_texts.append(extra_page_ocr)

            merged_lines: List[str] = []
            for blob in [ocr_text] + region_texts + [erp_text] + crop_ocr_texts:
                if blob:
                    merged_lines.extend([ln.strip() for ln in blob.splitlines() if ln.strip()])
            merged_lines = dedup_similar_lines(merged_lines, threshold=0.92)
            ocr_text = clean_for_embedding("\n".join(merged_lines[:80]))

        page_kind = classify_page_kind(body_text, ocr_text, red_regions)
        menu_path = infer_menu_path("\n".join([body_text, ocr_text, erp_text]))
        screen_title = derive_screen_title(body_text, menu_path, erp_text or ocr_text)

        visual = build_visual_summary(
            body_text=body_text,
            ocr_text=ocr_text,
            red_regions=red_regions,
            crop_ocr_texts=crop_ocr_texts,
            page_kind=page_kind,
            erp_detected=erp_detected,
            erp_text=erp_text,
        )
        retrieval_text = build_retrieval_text(
            page_num=i,
            body_text=body_text,
            ocr_text=ocr_text,
            visual_text=visual,
            menu_path=menu_path,
            page_kind=page_kind,
            screen_title=screen_title,
            erp_text=erp_text,
            red_regions=red_regions,
        )

        page_extra = {
            "menu_path": menu_path,
            "page_kind": page_kind,
            "red_regions": red_regions,
            "red_ratio": red_ratio,
            "has_red_box": bool(red_regions > 0),
            "has_step_numbers": bool(extract_step_numbers("\n".join([body_text, ocr_text, erp_text]))),
            "screen_title": screen_title,
            "erp_detected": erp_detected,
            "has_erp_screen": erp_detected,
            "erp_text_len": len(erp_text or ""),
            "actions": ", ".join(extract_actions("\n".join([body_text, ocr_text, erp_text]))),
            "field_keywords": ", ".join(extract_field_keywords("\n".join([body_text, ocr_text, erp_text]))),
        }

        if is_meaningful_text(body_text):
            body_sections.append(Section("body_text", body_text, f"{file_path.name} - page {i}", page_num=i, extra=page_extra))
        if enable_ocr and keep_ocr_text(ocr_text, min_len=4, min_score=0.08):
            ocr_sections.append(Section("ocr_text", ocr_text, f"{file_path.name} - page {i} OCR", page_num=i, extra=page_extra))
        if is_meaningful_text(visual, min_len=12):
            visual_sections.append(Section("visual_summary", visual, f"{file_path.name} - page {i} visual summary", page_num=i, extra=page_extra))
        if is_meaningful_text(retrieval_text, min_len=20):
            retrieval_sections.append(Section("retrieval_text", retrieval_text, f"{file_path.name} - page {i} retrieval", page_num=i, extra=page_extra))

    doc.close()
    return ExtractedDocument(
        file_path=file_path,
        file_type="pdf",
        body_sections=body_sections,
        ocr_sections=ocr_sections,
        visual_sections=visual_sections,
        retrieval_sections=retrieval_sections,
        meta=meta,
        original_filename=resolved_original,
    )


def make_chunks_for_document(doc: ExtractedDocument) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    all_sections = doc.body_sections + doc.ocr_sections + doc.visual_sections + doc.retrieval_sections
    embedded_at = time.strftime("%Y-%m-%d %H:%M:%S")

    for section in all_sections:
        if section.source_type == "body_text":
            pieces = split_text(section.text, max_chars=1400, overlap=180)
        elif section.source_type == "ocr_text":
            pieces = split_text(section.text, max_chars=700, overlap=70)
        elif section.source_type == "visual_summary":
            pieces = split_text(section.text, max_chars=650, overlap=50)
        else:
            pieces = split_text(section.text, max_chars=800, overlap=70)

        for idx, piece in enumerate(pieces, start=1):
            base_id = f"{doc.file_path.name}:{section.source_type}:{section.page_num or 0}:{idx}:{sha1_text(piece)[:12]}"
            meta = {
                "filename": doc.file_path.name,
                "original_filename": doc.original_filename,
                "created_at": embedded_at,
                "embedded_at": embedded_at,
                "source_path": str(doc.file_path),
                "page": section.page_num,
                "section_type": section.source_type,
                "file_name": doc.file_path.name,
                "file_path": str(doc.file_path),
                "page_num": section.page_num,
                "file_type": doc.file_type,
                "section_title": section.title,
                "chunk_index": idx,
            }
            if section.extra:
                meta.update(section.extra)
            out.append({"id": base_id, "text": piece, "metadata": meta})
    return out


def _embed_replace_same_original_enabled() -> bool:
    raw = (os.environ.get("GSLLM_EMBED_REPLACE_SAME_ORIGINAL", "1") or "").strip().lower()
    return raw not in ("0", "false", "no", "")


def _close_persistent_chroma_client(client: Any) -> None:
    if client is None:
        return
    try:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def upsert_to_chroma(
    chunks: List[Dict[str, Any]],
    chroma_dir: Path,
    collection_name: str,
    embed_model: str,
    batch_size: int = 64,
    *,
    purge_original_filename: Optional[str] = None,
) -> None:
    """
    단일 PersistentClient 로 purge(선택) + upsert 를 수행한다.
    클라이언트를 닫았다가 다시 열면 Chroma 에서 Component not running 이 날 수 있어 한 세션으로 묶는다.
    """
    ensure_dir(chroma_dir)
    print(f"[INFO] 임베딩 모델 로드: {embed_model}")
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    print(f"[INFO] 임베딩 디바이스: {device}")
    embedder = SentenceTransformer(embed_model, device=device)

    client = None
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        on = str(purge_original_filename or "").strip()
        if on and _embed_replace_same_original_enabled():
            try:
                col_names = [c.name for c in client.list_collections()]
                if collection_name in col_names:
                    col = client.get_collection(collection_name)
                    col.delete(where={"original_filename": {"$eq": on}})
            except Exception as e:
                print(
                    f"[PURGE_SAME_ORIGINAL_WARN] chroma_dir={chroma_dir} coll={collection_name} err={e!r}"
                )
        collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [x["text"] for x in batch]
            metas = [x["metadata"] for x in batch]
            ids = [x["id"] for x in batch]
            embs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()
            collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)
            print(f"  [EMBED] {min(start + batch_size, len(chunks))}/{len(chunks)}")
    finally:
        _close_persistent_chroma_client(client)


def discover_input_files(inputs: Sequence[str], recursive: bool) -> List[Path]:
    exts = {".pdf"}
    found: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix.lower() in exts:
            found.append(p.resolve())
        elif p.is_dir():
            iterator = p.rglob("*") if recursive else p.glob("*")
            for child in iterator:
                if child.is_file() and child.suffix.lower() in exts:
                    found.append(child.resolve())
    return sorted(set(found))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="embed_docs.py - paddleocr dynamic init version")
    p.add_argument("--inputs", nargs="+", default=[str(DEFAULT_INPUT_DIR)], help="입력 파일/폴더")
    p.add_argument("--recursive", action="store_true", default=True, help="하위폴더 재귀 탐색")
    p.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    p.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--enable-ocr", action="store_true", default=True)
    p.add_argument("--ocr-lang", default="kor+eng")
    p.add_argument("--save-page-images", action="store_true", help="렌더링 페이지 이미지를 저장")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--original-filename",
        default=None,
        dest="original_filename",
        help="업로드 원본 파일명(단일 입력 시 청크 metadata.original_filename)",
    )
    return p.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(levelname)s] %(message)s")


def _format_run_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}초"
    m, s = divmod(seconds, 60.0)
    if m < 60:
        return f"{int(m)}분 {s:.1f}초"
    h, m2 = divmod(m, 60.0)
    return f"{int(h)}시간 {int(m2)}분 {s:.1f}초"


def _print_run_timing(t0: float, start_wall: str) -> None:
    elapsed = time.perf_counter() - t0
    end_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[TIME] 시작 {start_wall} → 종료 {end_wall} | 총 소요 {_format_run_duration(elapsed)}")


def run_embed_job(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    """
    CLI `parse_args()`와 동일한 Namespace로 임베딩 파이프라인 실행.
    호출 전에 `configure_logging(args.verbose)`를 호출하는 것을 권장.
    반환: (exit_code, summary) — summary는 JSON 응답에 실을 수 있는 dict.
    """
    work_dir = ensure_dir(Path(args.work_dir))
    chroma_dir = ensure_dir(Path(args.chroma_dir))
    page_img_root = ensure_dir(work_dir / "page_images") if args.save_page_images else None

    engine_name, ocr_device = init_ocr_engine(use_gpu=True)
    print(f"[INFO] OCR 엔진: {engine_name}")
    print(f"[INFO] OCR 디바이스: {ocr_device}")
    if engine_name != "paddleocr":
        print("[WARN] PaddleOCR GPU 초기화에 실패했거나 미설치 상태라 Tesseract CPU fallback으로 동작합니다.")
        print("[WARN] GPU OCR을 원하면 paddleocr + paddlepaddle-gpu 설치 상태를 확인하세요.")

    input_files = discover_input_files(args.inputs, recursive=args.recursive)
    if not input_files:
        print("[ERROR] 입력 문서를 찾지 못했습니다.")
        return 1, {
            "ok": False,
            "error": "no_pdf_input",
            "message": "입력 경로에서 PDF를 찾지 못했습니다.",
            "inputs": list(args.inputs),
        }

    print("[INFO] 입력 문서")
    for x in input_files:
        print(f"  - {x}")
    print(f"[INFO] 총 {len(input_files)}개 문서")

    all_chunks: List[Dict[str, Any]] = []
    manifest_docs: List[Dict[str, Any]] = []
    total_body = total_ocr = total_visual = total_retrieval = 0

    n_inputs = len(input_files)
    explicit_orig = getattr(args, "original_filename", None)
    explicit_orig = (str(explicit_orig).strip() if explicit_orig else None)

    for idx, file_path in enumerate(input_files, start=1):
        print(f"\n[{idx}/{len(input_files)}] 처리: {file_path.name}")
        save_dir = page_img_root / file_path.stem if page_img_root else None

        per_original = explicit_orig if (n_inputs == 1 and explicit_orig) else None

        extracted = extract_pdf_document(
            file_path=file_path,
            enable_ocr=args.enable_ocr,
            ocr_lang=args.ocr_lang,
            save_page_images_dir=save_dir,
            original_filename=per_original,
        )
        chunks = make_chunks_for_document(extracted)
        doc_key = str(uuid.uuid4())
        for ch in chunks:
            ch.setdefault("metadata", {})
            ch["metadata"]["doc_key"] = doc_key
        all_chunks.extend(chunks)

        body_count = len(extracted.body_sections)
        ocr_count = len(extracted.ocr_sections)
        visual_count = len(extracted.visual_sections)
        retrieval_count = len(extracted.retrieval_sections)
        total_body += body_count
        total_ocr += ocr_count
        total_visual += visual_count
        total_retrieval += retrieval_count

        manifest_docs.append({
            "filename": file_path.name,
            "original_filename": extracted.original_filename,
            "doc_key": doc_key,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_path": str(file_path),
            "file_type": extracted.file_type,
            "pages": extracted.meta.get("pages", 0),
            "ocr_engine": extracted.meta.get("ocr_engine", OCR_ENGINE_NAME),
            "body_sections": body_count,
            "ocr_sections": ocr_count,
            "visual_sections": visual_count,
            "retrieval_sections": retrieval_count,
            "chunks": len(chunks),
        })
        print(f"  ✓ body={body_count} | ocr={ocr_count} | visual={visual_count} | retrieval={retrieval_count} | chunks={len(chunks)}")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "docs": manifest_docs,
        "summary": {
            "docs": len(manifest_docs),
            "body": total_body,
            "ocr": total_ocr,
            "visual": total_visual,
            "retrieval": total_retrieval,
            "chunks": len(all_chunks),
        },
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[SUMMARY]")
    print(f"  - docs:      {len(manifest_docs)}")
    print(f"  - body:      {total_body}")
    print(f"  - ocr:       {total_ocr}")
    print(f"  - visual:    {total_visual}")
    print(f"  - retrieval: {total_retrieval}")
    print(f"  - chunks:    {len(all_chunks)}")
    print(f"  - manifest:  {manifest_path}")

    print(f"\n[INFO] Chroma 업서트: {chroma_dir} / collection={args.collection}")
    chroma_lock = getattr(args, "chroma_lock", None)
    if chroma_lock is not None:
        chroma_lock.acquire()
    try:
        purge_orig: Optional[str] = None
        if _embed_replace_same_original_enabled() and len(input_files) == 1:
            purge_orig = explicit_orig if explicit_orig else input_files[0].name
            print(f"[INFO] 동일 original_filename 청크 purge 후 업서트(단일 Chroma 클라이언트): {purge_orig!r}")
        upsert_to_chroma(
            chunks=all_chunks,
            chroma_dir=chroma_dir,
            collection_name=args.collection,
            embed_model=args.embed_model,
            purge_original_filename=purge_orig,
        )
        print("[DONE] 임베딩 완료")
    finally:
        if chroma_lock is not None:
            chroma_lock.release()

    summary: Dict[str, Any] = {
        "ok": True,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "work_dir": str(work_dir),
        "chroma_dir": str(chroma_dir),
        "collection": args.collection,
        "embed_model": args.embed_model,
        "input_files": [str(p) for p in input_files],
    }
    return 0, summary


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    code, _ = run_embed_job(args)
    return code


if __name__ == "__main__":
    _run_t0 = time.perf_counter()
    _run_start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    atexit.register(lambda: _print_run_timing(_run_t0, _run_start_wall))
    sys.exit(main())
