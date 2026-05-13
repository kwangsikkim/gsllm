#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A안: 현재 코드 최대한 유지형 embed_docs.py
- Upstage 미사용
- PDF 중심
- 기존 구조 유지
- body_text / ocr_text / visual_summary 분리 임베딩
- Tesseract 유지
- 개선점
  1) 페이지 전체 OCR + 상/중/하 분할 OCR + 빨간영역 crop OCR
  2) 영역별 PSM 분기
  3) OCR 필터 완화
  4) visual_summary 강화
  5) embed_query_all.py 호환 메타 저장
- 비약 OCR(--ocr-hyper, 기본 ON)
  - OpenCV: deskew + 전처리 variant(adaptive / Otsu / 비이진)
  - Tesseract: 다중 PSM 후보 중 품질점수+신뢰도로 선택
  - 선택: PaddleOCR 설치 시 병합(--paddle-ocr)
     - filename, source_path, page, section_type
     - legacy alias: file_name, file_path, page_num

기본 경로:
- 입력:      /home/siwasoft/gsllm/embed_test/data/IT_file
- work:      /home/siwasoft/gsllm/embed_test/work/collection_A
- chroma:    /home/siwasoft/gsllm/embed_test/chroma/collection_A
- collection: collection_A
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from sentence_transformers import SentenceTransformer

try:
    import cv2
except Exception:
    cv2 = None

try:
    from paddleocr import PaddleOCR as _PaddleOCRClass  # type: ignore
except Exception:
    _PaddleOCRClass = None

LOG = logging.getLogger("embed_docs")

# Paddle lazy singleton (무거움)
_PADDLE_OCR: Any = None
_PADDLE_INIT_FAILED = False

DEFAULT_INPUT_DIR = Path("/home/siwasoft/gsllm/embed_test/data/IT_file")
DEFAULT_WORK_DIR = Path("/home/siwasoft/gsllm/embed_test/work/collection_A")
DEFAULT_CHROMA_DIR = Path("/home/siwasoft/gsllm/embed_test/chroma/collection_A")
DEFAULT_COLLECTION = "collection_A"
DEFAULT_EMBED_MODEL = "/home/siwasoft/bge-m3"


@dataclass
class Section:
    source_type: str   # body_text | ocr_text | visual_summary
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
    meta: Dict[str, Any]


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


def clean_for_embedding(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    text = re.sub(r"\s*→\s*", " → ", text)
    text = re.sub(r"\s*•\s*", " • ", text)
    text = re.sub(r"\s*·\s*", " · ", text)
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
    weird = len(re.findall(r"[^가-힣A-Za-z0-9\s\-\_/.,:()%\[\]→•·+]", s))
    spaces = s.count(" ")
    score = 0.0
    score += min(1.0, useful / max(1, total)) * 0.42
    score += min(1.0, hangul / max(1, total * 0.22)) * 0.18
    score += max(0.0, 1.0 - weird / max(1, total)) * 0.20
    score += min(1.0, spaces / max(1, total / 14.0)) * 0.20
    return round(score, 4)


def keep_ocr_text(text: str, *, min_len: int = 4, min_score: float = 0.08) -> bool:
    s = clean_for_embedding(text)
    if len(s) < min_len:
        return False
    if re.search(r"(조회|신규|수정|저장|입력|선택|확인|POS|SCM|코드|관리|필수)", s, flags=re.I):
        return True
    return ocr_quality_score(s) >= min_score


def infer_menu_path(text: str) -> str:
    text = clean_for_embedding(text)
    patterns = [
        r"(설정\s*[–-]\s*[가-힣A-Za-z0-9 ]+\s*[–-]\s*[가-힣A-Za-z0-9 ()/]+\s*(?:[–-]\s*[가-힣A-Za-z0-9 ()/]+)?)",
        r"(물류\s*관리\s*[–-]\s*[가-힣A-Za-z0-9 ]+\s*[–-]\s*[가-힣A-Za-z0-9 ()/]+\s*(?:[–-]\s*[가-힣A-Za-z0-9 ()/]+)?)",
        r"([가-힣A-Za-z0-9 ]+\s*[–-]\s*[가-힣A-Za-z0-9 ]+\s*[–-]\s*[가-힣A-Za-z0-9 ()/]+\s*(?:[–-]\s*[가-힣A-Za-z0-9 ()/]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return re.sub(r"\s*[–-]\s*", " > ", m.group(1)).strip()
    return ""


def extract_step_numbers(text: str) -> List[str]:
    text = clean_for_embedding(text)
    found = re.findall(
        r"(?:^|\s)([1-9][0-9]?)\s*(?:[.)]|)(?=\s*(?:조회|신규|수정|저장|클릭|입력|선택|확인))",
        text,
    )
    out: List[str] = []
    for x in found:
        if x not in out:
            out.append(x)
    return out[:10]


def render_pdf_page(page: Any, dpi: int = 190) -> Optional[Image.Image]:
    try:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None


def preprocess_for_ocr(img: Image.Image, scale: float = 2.4, threshold: bool = True) -> Image.Image:
    w, h = img.size
    img = img.convert("L")
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)

    if cv2 is not None and threshold:
        arr = np.array(img)
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
        arr = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        img = Image.fromarray(arr)
    return img


def run_ocr(img: Image.Image, lang: str = "kor+eng", psm: int = 6) -> str:
    try:
        text = pytesseract.image_to_string(img, lang=lang, config=f"--oem 3 --psm {psm}")
    except Exception:
        return ""
    return clean_for_embedding(text)


def deskew_rgb(img: Image.Image) -> Image.Image:
    """OpenCV로 소각도 보정. 실패 시 원본."""
    if cv2 is None:
        return img
    try:
        rgb = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        if coords.size < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.25:
            return img
        if abs(angle) > 12:
            return img
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    except Exception:
        return img


def preprocess_variant(
    img: Image.Image,
    *,
    scale: float,
    threshold: bool,
    mode: str = "adaptive",
) -> Image.Image:
    """
    mode: adaptive (기존 Gaussian adaptive), otsu, none (대비·샤프만)
    """
    w, h = img.size
    g = img.convert("L")
    g = g.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    g = ImageEnhance.Contrast(g).enhance(2.0)
    g = g.filter(ImageFilter.SHARPEN)

    if not threshold:
        return g

    if cv2 is None:
        return g

    arr = np.array(g)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    if mode == "otsu":
        _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        arr = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
    return Image.fromarray(arr)


def tesseract_mean_confidence(img: Image.Image, lang: str, psm: int) -> float:
    try:
        data = pytesseract.image_to_data(
            img,
            lang=lang,
            config=f"--oem 3 --psm {psm}",
            output_type=pytesseract.Output.DICT,
        )
        confs: List[int] = []
        for c in data.get("conf", []):
            try:
                v = int(float(c))
            except (TypeError, ValueError):
                continue
            if v > 0:
                confs.append(v)
        if not confs:
            return 0.0
        return sum(confs) / len(confs) / 100.0
    except Exception:
        return 0.0


def combined_ocr_score(text: str, mean_conf: float) -> float:
    q = ocr_quality_score(text)
    ln = min(0.22, len(clean_for_embedding(text)) / 9000.0)
    return q * 0.72 + mean_conf * 0.22 + ln


def best_tesseract_ocr(
    img_rgb: Image.Image,
    lang: str,
    *,
    psms: Sequence[int],
    max_calls: int = 24,
) -> str:
    """여러 전처리 × PSM 조합 중 combined_ocr_score 최고인 텍스트."""
    img_work = deskew_rgb(img_rgb)
    variants: List[Tuple[str, float, bool, str]] = [
        ("v1", 2.2, True, "adaptive"),
        ("v2", 2.6, True, "adaptive"),
        ("v3", 2.4, False, "none"),
    ]
    if cv2 is not None:
        variants.append(("v4", 2.8, True, "otsu"))

    best_text = ""
    best_score = -1.0
    calls = 0
    for _name, scale, use_th, mode in variants:
        proc = preprocess_variant(img_work, scale=scale, threshold=use_th, mode=mode)
        for psm in psms:
            if calls >= max_calls:
                break
            calls += 1
            txt = run_ocr(proc, lang=lang, psm=int(psm))
            conf = tesseract_mean_confidence(proc, lang, int(psm))
            sc = combined_ocr_score(txt, conf)
            if sc > best_score and txt.strip():
                best_score = sc
                best_text = txt
        if calls >= max_calls:
            break
    return best_text


def get_paddle_ocr() -> Any:
    global _PADDLE_OCR, _PADDLE_INIT_FAILED
    if _PADDLE_INIT_FAILED:
        return None
    if _PaddleOCRClass is None:
        _PADDLE_INIT_FAILED = True
        return None
    if _PADDLE_OCR is None:
        try:
            kwargs = dict(use_angle_cls=True, lang="korean")
            try:
                _PADDLE_OCR = _PaddleOCRClass(**kwargs, show_log=False)  # type: ignore[call-arg]
            except TypeError:
                _PADDLE_OCR = _PaddleOCRClass(**kwargs)
        except Exception as e:
            LOG.debug("PaddleOCR init failed: %s", e)
            _PADDLE_INIT_FAILED = True
            return None
    return _PADDLE_OCR


def run_paddle_lines(img_rgb: Image.Image) -> str:
    ocr = get_paddle_ocr()
    if ocr is None:
        return ""
    try:
        arr = np.array(img_rgb.convert("RGB"))
        out = ocr.ocr(arr, cls=True)
        lines: List[str] = []
        if not out or out[0] is None:
            return ""
        for row in out[0]:
            if row is None or len(row) < 2:
                continue
            piece = row[1]
            if isinstance(piece, (list, tuple)) and piece:
                t = str(piece[0]).strip()
            else:
                t = str(piece).strip()
            if t:
                lines.append(t)
        return clean_for_embedding("\n".join(lines))
    except Exception as e:
        LOG.debug("Paddle OCR run failed: %s", e)
        return ""


def fuse_ocr_texts(*parts: str) -> str:
    """여러 OCR 결과를 줄 단위 dedupe 병합."""
    seen = set()
    out_lines: List[str] = []
    for p in parts:
        s = clean_for_embedding(p or "")
        if not s:
            continue
        for line in s.splitlines():
            t = line.strip()
            if not t:
                continue
            key = re.sub(r"\s+", "", t)
            if key in seen:
                continue
            seen.add(key)
            out_lines.append(t)
    return clean_for_embedding("\n".join(out_lines))


def ocr_hyper_full_page(
    img: Image.Image,
    lang: str,
    *,
    use_paddle: bool,
    max_tesseract_calls: int = 24,
) -> str:
    tess = best_tesseract_ocr(
        img,
        lang,
        psms=(6, 11, 3, 4, 12),
        max_calls=max_tesseract_calls,
    )
    if not use_paddle:
        return tess
    pad = run_paddle_lines(img)
    if not pad.strip():
        return tess
    if not tess.strip():
        return pad
    st = ocr_quality_score(tess)
    sp = ocr_quality_score(pad)
    if sp > st + 0.03:
        return fuse_ocr_texts(pad, tess)
    return fuse_ocr_texts(tess, pad)


def ocr_hyper_region(
    img: Image.Image,
    lang: str,
    psm: int,
    *,
    use_paddle: bool,
    max_calls: int = 10,
) -> str:
    """작은 영역: 전처리 subset + PSM 고정/소량."""
    img_work = deskew_rgb(img)
    best = ""
    best_sc = -1.0
    calls = 0
    for scale, use_th, mode in ((2.4, True, "adaptive"), (2.7, True, "adaptive"), (2.3, False, "none")):
        if calls >= max_calls:
            break
        proc = preprocess_variant(img_work, scale=scale, threshold=use_th, mode=mode)
        for p in (psm, 6, 11):
            if calls >= max_calls:
                break
            calls += 1
            txt = run_ocr(proc, lang=lang, psm=int(p))
            conf = tesseract_mean_confidence(proc, lang, int(p))
            sc = combined_ocr_score(txt, conf)
            if sc > best_sc and txt.strip():
                best_sc = sc
                best = txt
    if use_paddle:
        pad = run_paddle_lines(img)
        if pad and combined_ocr_score(pad, 0.5) > best_sc:
            best = fuse_ocr_texts(best, pad) if best else pad
        elif pad:
            best = fuse_ocr_texts(best, pad)
    return best


def detect_red_regions(img: Image.Image) -> Tuple[int, float, List[Tuple[int, int, int, int]]]:
    if cv2 is None:
        return 0, 0.0, []

    rgb = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower1 = np.array([0, 28, 35], dtype=np.uint8)
    upper1 = np.array([18, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 28, 35], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    H, W = mask.shape[:2]
    total_pixels = float(H * W)
    red_ratio = float(np.count_nonzero(mask)) / max(1.0, total_pixels)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(80, int(W * H * 0.00008))
    boxes: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        if w < 8 or h < 6:
            continue
        if w > int(W * 0.98) and h < int(H * 0.08):
            continue
        boxes.append((x, y, x + w, y + h))

    boxes = sorted(boxes, key=lambda b: ((b[1] + b[3]) // 2, (b[0] + b[2]) // 2))

    merged: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        hit = False
        for i, (ax1, ay1, ax2, ay2) in enumerate(merged):
            if not (x2 < ax1 - 16 or x1 > ax2 + 16 or y2 < ay1 - 16 or y1 > ay2 + 16):
                merged[i] = (min(x1, ax1), min(y1, ay1), max(x2, ax2), max(y2, ay2))
                hit = True
                break
        if not hit:
            merged.append(box)

    return len(merged), round(red_ratio, 6), merged[:16]


def crop_with_padding(img: Image.Image, box: Tuple[int, int, int, int], pad: int = 24) -> Image.Image:
    x1, y1, x2, y2 = box
    return img.crop(
        (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(img.width, x2 + pad),
            min(img.height, y2 + pad),
        )
    )


def split_vertical_regions(img: Image.Image) -> List[Tuple[str, Image.Image, int]]:
    """
    현재 코드 유지형 보강:
    - 상단 / 중단 / 하단 분할 OCR
    - 영역별로 다른 PSM 사용
    """
    w, h = img.size
    top = img.crop((0, 0, w, int(h * 0.23)))
    mid = img.crop((0, int(h * 0.18), w, int(h * 0.82)))
    bot = img.crop((0, int(h * 0.74), w, h))
    return [
        ("top_region", top, 6),
        ("mid_region", mid, 6),
        ("bottom_region", bot, 7),
    ]


def classify_page_kind(body_text: str, ocr_text: str, red_regions: int) -> str:
    combined = clean_for_embedding("\n".join([body_text or "", ocr_text or ""]))
    if red_regions >= 1 and re.search(r"(조회|신규|수정|저장|클릭|입력|선택)", combined):
        return "manual_ui"
    if re.search(r"(메뉴위치|메뉴 위치|조회 클릭|신규 클릭|수정 클릭|저장 클릭)", combined):
        return "manual_ui"
    if re.search(r"(contents|보고서|현황|수불추적|재고)", combined, flags=re.I):
        return "report_or_table"
    return "general"


def build_visual_summary(
    *,
    body_text: str,
    ocr_text: str,
    red_regions: int,
    red_ratio: float,
    crop_ocr_texts: Sequence[str],
    page_kind: str,
) -> str:
    body_text = clean_for_embedding(body_text)
    ocr_text = clean_for_embedding(ocr_text)
    combined = "\n".join([body_text, ocr_text] + [clean_for_embedding(x) for x in crop_ocr_texts if x])

    menu_path = infer_menu_path(combined)
    step_numbers = extract_step_numbers(combined)
    has_required = ("필수 입력" in combined) or ("필수값" in combined) or ("표시는 필수" in combined)
    has_pos = ("POS" in combined.upper()) or ("SCM" in combined.upper())
    has_save = ("저장" in combined)
    has_query_new = ("조회" in combined and "신규" in combined)

    bits: List[str] = []
    if menu_path:
        bits.append(f"메뉴 경로는 {menu_path} 입니다.")
    if page_kind == "manual_ui":
        bits.append("이 페이지는 화면 사용법을 설명하는 매뉴얼형 UI 페이지로 보입니다.")
    if step_numbers:
        bits.append(f"페이지에는 단계 번호 {', '.join(step_numbers)} 가 나타납니다.")
    elif has_query_new and has_save:
        bits.append("페이지는 조회 → 신규 → 저장 순서의 작업 절차를 설명하는 것으로 보입니다.")
    if red_regions > 0:
        bits.append(f"빨간 강조 영역이 {red_regions}개 감지되었습니다.")
        bits.append("빨간 박스 또는 빨간 표시가 있는 영역은 사용자가 주의해서 입력하거나 확인해야 하는 영역일 가능성이 높습니다.")
    if has_required:
        bits.append("설명에는 필수 입력 값 안내가 포함됩니다.")
    if has_pos:
        bits.append("설명에는 POS 또는 SCM 연결 정보 입력 안내가 포함됩니다.")

    useful_crops = [x for x in crop_ocr_texts if keep_ocr_text(x)]
    if useful_crops:
        bits.append(f"강조 영역 주변 OCR 요약 : {' | '.join(useful_crops[:3])[:450]}")

    if not bits:
        bits.append("이 페이지는 문서 본문 중심 페이지입니다.")

    return clean_for_embedding(" ".join(bits))


def extract_pdf_document(
    file_path: Path,
    enable_ocr: bool,
    ocr_lang: str,
    save_page_images_dir: Optional[Path] = None,
    *,
    ocr_hyper: bool = True,
    use_paddle_ocr: bool = False,
    ocr_max_calls_full: int = 24,
    ocr_max_calls_region: int = 10,
) -> ExtractedDocument:
    body_sections: List[Section] = []
    ocr_sections: List[Section] = []
    visual_sections: List[Section] = []
    meta: Dict[str, Any] = {"pages": 0, "renderer": "fitz"}

    try:
        doc = fitz.open(str(file_path))
    except Exception as e:
        raise RuntimeError(f"PDF open failed: {e}") from e

    meta["pages"] = len(doc)

    for i, page in enumerate(doc, start=1):
        body_text = clean_for_embedding(page.get_text("text") or "")
        img = render_pdf_page(page)

        if save_page_images_dir and img is not None:
            ensure_dir(save_page_images_dir)
            img.save(save_page_images_dir / f"page_{i:04d}.png")

        ocr_text = ""
        crop_ocr_texts: List[str] = []
        red_regions = 0
        red_ratio = 0.0

        if enable_ocr and img is not None:
            if ocr_hyper:
                # 1) full page: multi-preprocess × multi-PSM (+ optional Paddle)
                base_ocr = ocr_hyper_full_page(
                    img,
                    ocr_lang,
                    use_paddle=use_paddle_ocr,
                    max_tesseract_calls=ocr_max_calls_full,
                )
                if keep_ocr_text(base_ocr, min_len=4, min_score=0.08):
                    ocr_text = base_ocr

                # 2) top / mid / bottom
                region_texts = []
                for _region_name, region_img, region_psm in split_vertical_regions(img):
                    reg_ocr = ocr_hyper_region(
                        region_img,
                        ocr_lang,
                        region_psm,
                        use_paddle=use_paddle_ocr,
                        max_calls=ocr_max_calls_region,
                    )
                    if keep_ocr_text(reg_ocr, min_len=4, min_score=0.06):
                        region_texts.append(reg_ocr)

                # 3) red crops
                red_regions, red_ratio, boxes = detect_red_regions(img)
                for box in boxes[:10]:
                    crop = crop_with_padding(img, box, pad=24)
                    crop_ocr = ocr_hyper_region(
                        crop,
                        ocr_lang,
                        6,
                        use_paddle=use_paddle_ocr,
                        max_calls=min(8, ocr_max_calls_region),
                    )
                    if keep_ocr_text(crop_ocr, min_len=3, min_score=0.05):
                        crop_ocr_texts.append(crop_ocr)
            else:
                # legacy 단일 경로
                base_ocr = run_ocr(preprocess_for_ocr(img, scale=2.2, threshold=True), lang=ocr_lang, psm=6)
                if keep_ocr_text(base_ocr, min_len=4, min_score=0.08):
                    ocr_text = base_ocr

                region_texts = []
                for _region_name, region_img, region_psm in split_vertical_regions(img):
                    reg_ocr = run_ocr(
                        preprocess_for_ocr(region_img, scale=2.5, threshold=True),
                        lang=ocr_lang,
                        psm=region_psm,
                    )
                    if keep_ocr_text(reg_ocr, min_len=4, min_score=0.06):
                        region_texts.append(reg_ocr)

                red_regions, red_ratio, boxes = detect_red_regions(img)
                for box in boxes[:10]:
                    crop = crop_with_padding(img, box, pad=24)
                    crop_ocr = run_ocr(
                        preprocess_for_ocr(crop, scale=2.8, threshold=True),
                        lang=ocr_lang,
                        psm=6,
                    )
                    if keep_ocr_text(crop_ocr, min_len=3, min_score=0.05):
                        crop_ocr_texts.append(crop_ocr)

            # merge OCR text but keep separate section
            merged_ocr = clean_for_embedding("\n".join([ocr_text] + region_texts + crop_ocr_texts))
            # de-dup roughly line based
            dedup_lines: List[str] = []
            seen = set()
            for line in merged_ocr.splitlines():
                s = line.strip()
                if not s:
                    continue
                key = re.sub(r"\s+", "", s)
                if key in seen:
                    continue
                seen.add(key)
                dedup_lines.append(s)
            ocr_text = clean_for_embedding("\n".join(dedup_lines))

        page_kind = classify_page_kind(body_text, ocr_text, red_regions)
        menu_path = infer_menu_path("\n".join([body_text, ocr_text]))
        visual = build_visual_summary(
            body_text=body_text,
            ocr_text=ocr_text,
            red_regions=red_regions,
            red_ratio=red_ratio,
            crop_ocr_texts=crop_ocr_texts,
            page_kind=page_kind,
        )

        page_extra = {
            "menu_path": menu_path,
            "page_kind": page_kind,
            "red_regions": red_regions,
            "red_ratio": red_ratio,
            "has_red_box": bool(red_regions > 0),
            "has_step_numbers": bool(extract_step_numbers("\n".join([body_text, ocr_text]))),
        }

        if is_meaningful_text(body_text):
            body_sections.append(
                Section(
                    "body_text",
                    body_text,
                    f"{file_path.name} - page {i}",
                    page_num=i,
                    extra=page_extra,
                )
            )

        if enable_ocr and keep_ocr_text(ocr_text, min_len=4, min_score=0.06):
            ocr_sections.append(
                Section(
                    "ocr_text",
                    ocr_text,
                    f"{file_path.name} - page {i} OCR",
                    page_num=i,
                    extra=page_extra,
                )
            )

        if is_meaningful_text(visual, min_len=10):
            visual_sections.append(
                Section(
                    "visual_summary",
                    visual,
                    f"{file_path.name} - page {i} visual summary",
                    page_num=i,
                    extra=page_extra,
                )
            )

    doc.close()

    return ExtractedDocument(
        file_path=file_path,
        file_type="pdf",
        body_sections=body_sections,
        ocr_sections=ocr_sections,
        visual_sections=visual_sections,
        meta=meta,
    )


def make_chunks_for_document(doc: ExtractedDocument) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    all_sections = doc.body_sections + doc.ocr_sections + doc.visual_sections
    for section in all_sections:
        if section.source_type == "body_text":
            pieces = split_text(section.text, max_chars=1400, overlap=180)
        elif section.source_type == "ocr_text":
            pieces = split_text(section.text, max_chars=900, overlap=90)
        else:
            pieces = split_text(section.text, max_chars=700, overlap=60)

        for idx, piece in enumerate(pieces, start=1):
            base_id = f"{doc.file_path.name}:{section.source_type}:{section.page_num or 0}:{idx}:{sha1_text(piece)[:12]}"
            meta = {
                # query compatibility
                "filename": doc.file_path.name,
                "source_path": str(doc.file_path),
                "page": section.page_num,
                "section_type": section.source_type,
                # legacy aliases
                "file_name": doc.file_path.name,
                "file_path": str(doc.file_path),
                "page_num": section.page_num,
                # additional metadata
                "file_type": doc.file_type,
                "section_title": section.title,
                "chunk_index": idx,
            }
            if section.extra:
                meta.update(section.extra)
            out.append({
                "id": base_id,
                "text": piece,
                "metadata": meta,
            })
    return out


def upsert_to_chroma(
    chunks: List[Dict[str, Any]],
    chroma_dir: Path,
    collection_name: str,
    embed_model: str,
    batch_size: int = 64,
) -> None:
    ensure_dir(chroma_dir)
    print(f"[INFO] 임베딩 모델 로드: {embed_model}")
    embedder = SentenceTransformer(embed_model)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [x["text"] for x in batch]
        metas = [x["metadata"] for x in batch]
        ids = [x["id"] for x in batch]
        embs = embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
        collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)


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
    p = argparse.ArgumentParser(description="embed_docs.py - A안(현재 코드 최대한 유지형)")
    p.add_argument("--inputs", nargs="+", default=[str(DEFAULT_INPUT_DIR)], help="입력 파일/폴더")
    p.add_argument("--recursive", action="store_true", default=True, help="하위폴더 재귀 탐색")
    p.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    p.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--enable-ocr", action="store_true", default=True)
    p.add_argument("--ocr-lang", default="kor+eng")
    p.add_argument(
        "--no-ocr-hyper",
        action="store_true",
        help="비약 OCR 비활성화 (단일 전처리+PSM 경로만 사용, 더 빠름)",
    )
    p.add_argument(
        "--paddle-ocr",
        action="store_true",
        help="PaddleOCR 결과를 Tesseract와 병합 (pip install paddleocr 필요)",
    )
    p.add_argument("--ocr-max-calls-full", type=int, default=24, help="페이지 전체 OCR 시 Tesseract 시도 상한")
    p.add_argument("--ocr-max-calls-region", type=int, default=10, help="영역/crop OCR 시 Tesseract 시도 상한")
    p.add_argument("--save-page-images", action="store_true", help="렌더링 페이지 이미지를 저장")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    work_dir = ensure_dir(Path(args.work_dir))
    chroma_dir = ensure_dir(Path(args.chroma_dir))
    page_img_root = ensure_dir(work_dir / "page_images") if args.save_page_images else None

    input_files = discover_input_files(args.inputs, recursive=args.recursive)
    if not input_files:
        print("[ERROR] 입력 문서를 찾지 못했습니다.")
        return 1

    print("[INFO] 입력 문서")
    for x in input_files:
        print(f"  - {x}")
    print(f"[INFO] 총 {len(input_files)}개 문서")

    all_chunks: List[Dict[str, Any]] = []
    manifest_docs: List[Dict[str, Any]] = []
    total_body = total_ocr = total_visual = 0

    for idx, file_path in enumerate(input_files, start=1):
        print(f"\n[{idx}/{len(input_files)}] 처리: {file_path.name}")
        save_dir = page_img_root / file_path.stem if page_img_root else None

        extracted = extract_pdf_document(
            file_path=file_path,
            enable_ocr=args.enable_ocr,
            ocr_lang=args.ocr_lang,
            save_page_images_dir=save_dir,
            ocr_hyper=not args.no_ocr_hyper,
            use_paddle_ocr=args.paddle_ocr,
            ocr_max_calls_full=args.ocr_max_calls_full,
            ocr_max_calls_region=args.ocr_max_calls_region,
        )
        chunks = make_chunks_for_document(extracted)
        all_chunks.extend(chunks)

        body_count = len(extracted.body_sections)
        ocr_count = len(extracted.ocr_sections)
        visual_count = len(extracted.visual_sections)
        total_body += body_count
        total_ocr += ocr_count
        total_visual += visual_count

        manifest_docs.append({
            "filename": file_path.name,
            "source_path": str(file_path),
            "file_type": extracted.file_type,
            "pages": extracted.meta.get("pages", 0),
            "body_sections": body_count,
            "ocr_sections": ocr_count,
            "visual_sections": visual_count,
            "chunks": len(chunks),
        })
        print(f"  ✓ body={body_count} | ocr={ocr_count} | visual={visual_count} | chunks={len(chunks)}")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "docs": manifest_docs,
        "summary": {
            "docs": len(manifest_docs),
            "body": total_body,
            "ocr": total_ocr,
            "visual": total_visual,
            "chunks": len(all_chunks),
        },
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[SUMMARY]")
    print(f"  - docs:    {len(manifest_docs)}")
    print(f"  - body:    {total_body}")
    print(f"  - ocr:     {total_ocr}")
    print(f"  - visual:  {total_visual}")
    print(f"  - chunks:  {len(all_chunks)}")
    print(f"  - manifest: {manifest_path}")

    print(f"\n[INFO] Chroma 업서트: {chroma_dir} / collection={args.collection}")
    upsert_to_chroma(
        chunks=all_chunks,
        chroma_dir=chroma_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
    )
    print("[DONE] 임베딩 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
