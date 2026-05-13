"""
매뉴얼 PDF 파일 임베딩 시스템
- PDF를 페이지별로 분할
- 이미지, 테이블 추출 및 저장 (이미지: PyMuPDF, 테이블: pdfplumber)
- 텍스트 블록 + bbox 추출 및 구조화
- 이미지/테이블 주변 local text 수집
- media_text 생성 (섹션/서브섹션 + local_text + 테이블 일부 등)
- BAAI/bge-m3 모델을 사용한 임베딩
  - main 컬렉션: 텍스트/이미지/테이블 검색용 임베딩
  - media 컬렉션: 이미지/테이블 의미 매칭용 임베딩 (media_text 기반)
- 여러 PDF를 처리해도 metadata.json이 덮어쓰기 되지 않도록 파일 단위로 누적 저장
"""

import os
import json
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import chromadb
import pdfplumber  # 테이블 추출용


EMBED_MODEL = "BAAI/bge-m3"


class ManualEmbedder:
    """매뉴얼 PDF 임베딩 클래스"""

    def __init__(self, output_dir: str = "manual_output", embed_model: str = EMBED_MODEL):
        """
        Args:
            output_dir: 출력 디렉토리 경로
            embed_model: 임베딩 모델 이름
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 이미지/테이블 저장 디렉토리
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"
        self.images_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Embedding device: {self.device}")

        # 임베딩 모델 로드
        print(f"Loading embedding model: {embed_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.embed_model = AutoModel.from_pretrained(embed_model)
        self.embed_model.to(self.device)
        self.embed_model.eval()

        # ChromaDB 초기화 (새 버전 방식)
        db_path = self.output_dir / "chroma_db"
        self.client = chromadb.PersistentClient(path=str(db_path))

        # 메인 컬렉션 (텍스트 + 이미지/테이블 일반 검색용)
        self.collection = self.client.get_or_create_collection(
            name="manual_embeddings",
            metadata={"hnsw:space": "cosine"},
        )

        # 미디어 전용 컬렉션 (이미지/테이블 의미 매칭용)
        self.media_collection = self.client.get_or_create_collection(
            name="manual_media_embeddings",
            metadata={"hnsw:space": "cosine"},
        )

        # 메타데이터 파일 경로
        self.metadata_file = self.output_dir / "metadata.json"

    # =====================================================================
    # 메타데이터 JSON 로드/저장 (여러 PDF 누적 저장)
    # =====================================================================

    def _load_metadata_all(self) -> Dict[str, Any]:
        """기존 metadata.json 전체 로드 (없으면 기본 구조 생성)"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        # 기본 구조 보정
        if "files" not in data or not isinstance(data.get("files"), list):
            data["files"] = []
        if "summary" not in data or not isinstance(data.get("summary"), dict):
            data["summary"] = {
                "total_files": 0,
                "total_pages": 0,
                "total_documents": 0,
            }
        return data

    def _save_metadata_all(self, data: Dict[str, Any]) -> None:
        """metadata.json 전체 저장"""
        files = data.get("files", [])
        total_files = len(files)
        total_pages = sum(f.get("total_pages", 0) for f in files)
        total_documents = sum(f.get("total_documents", 0) for f in files)

        data["summary"] = {
            "total_files": total_files,
            "total_pages": total_pages,
            "total_documents": total_documents,
        }

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # =====================================================================
    # 이미지 / 테이블 추출
    # =====================================================================

    def extract_images_and_tables(self, pdf_path: str) -> Dict[int, List[Dict[str, Any]]]:
        """
        PDF에서 이미지와 테이블을 추출하여 저장

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            페이지 번호를 키로 하는 이미지/테이블 정보 딕셔너리
            { page_num: [ { type, filename, path, bbox, index, text?, local_text? }, ... ], ... }
        """
        # PyMuPDF: 이미지 추출용
        doc = fitz.open(pdf_path)
        # pdfplumber: 테이블 추출용
        pdf = pdfplumber.open(pdf_path)

        page_elements: Dict[int, List[Dict[str, Any]]] = {}

        for page_num in range(len(doc)):
            page_index = page_num  # 0-based
            fitz_page = doc[page_index]
            plumber_page = pdf.pages[page_index]
            elements: List[Dict[str, Any]] = []

            # ======================
            # 1) 이미지 추출 (PyMuPDF)
            # ======================
            image_list = fitz_page.get_images(full=True)
            page_width = fitz_page.rect.width
            page_height = fitz_page.rect.height

            for img_idx, img in enumerate(image_list):
                xref = img[0]

                # 이미지 위치 정보 먼저 가져오기
                image_rects = fitz_page.get_image_rects(xref)
                if not image_rects:
                    continue

                rect = image_rects[0]

                # 사이드바 / 너무 작은 / 비율 이상한 이미지 필터링
                image_width = rect.x1 - rect.x0
                image_height = rect.y1 - rect.y0
                image_ratio = image_width / image_height if image_height > 0 else 0

                is_sidebar_like = False
                # 조건 1: 오른쪽 끝에 있고 좁은 너비
                if rect.x0 > page_width * 0.8 and image_width < page_width * 0.15:
                    is_sidebar_like = True
                # 조건 2: 이미지 비율이 비정상적 (너무 좁거나 너무 넓음)
                elif image_ratio < 0.1 or image_ratio > 10:
                    is_sidebar_like = True
                # 조건 3: 이미지가 너무 작음
                elif image_width < page_width * 0.1 and image_height < page_height * 0.1:
                    is_sidebar_like = True

                # ★ 사이드바/쓸모없는 이미지는 아예 스킵 (파일도 안 만들고, 메타에도 안 남김)
                if is_sidebar_like:
                    continue

                # 여기부터는 실제로 사용할 이미지만 처리
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_width_pixels = base_image["width"]
                image_height_pixels = base_image["height"]

                original_img = Image.open(io.BytesIO(image_bytes))

                # 페이지 경계 내로 클리핑
                clipped_rect = fitz.Rect(
                    max(0, rect.x0),
                    max(0, rect.y0),
                    min(page_width, rect.x1),
                    min(page_height, rect.y1),
                )

                image_rect_width = rect.x1 - rect.x0
                image_rect_height = rect.y1 - rect.y0

                left_offset = max(0, -rect.x0)
                top_offset = max(0, -rect.y0)
                right_offset = max(0, rect.x1 - page_width)
                bottom_offset = max(0, rect.y1 - page_height)

                scale_x = (
                    image_width_pixels / image_rect_width
                    if image_rect_width > 0
                    else 1.0
                )
                scale_y = (
                    image_height_pixels / image_rect_height
                    if image_rect_height > 0
                    else 1.0
                )

                crop_left = int(left_offset * scale_x)
                crop_top = int(top_offset * scale_y)
                crop_right = int(image_width_pixels - right_offset * scale_x)
                crop_bottom = int(image_height_pixels - bottom_offset * scale_y)

                cropped_img = original_img.crop(
                    (crop_left, crop_top, crop_right, crop_bottom)
                )

                image_filename = f"page_{page_index+1}_img_{img_idx+1}.png"
                image_path = self.images_dir / image_filename
                cropped_img.save(str(image_path))

                elements.append(
                    {
                        "type": "image",
                        "filename": image_filename,
                        "path": str(image_path),
                        "bbox": [
                            float(clipped_rect.x0),
                            float(clipped_rect.y0),
                            float(clipped_rect.x1),
                            float(clipped_rect.y1),
                        ],
                        "index": img_idx + 1,
                        # local_text, media_text는 나중에 붙임
                    }
                )

            # ======================
            # 2) 테이블 추출 (pdfplumber)
            # ======================
            tables = plumber_page.find_tables()

            page_width = plumber_page.width
            page_height = plumber_page.height

            for table_idx, table in enumerate(tables):
                bbox = table.bbox  # (x0, top, x1, bottom)

                clipped_bbox = (
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(page_width, bbox[2]),
                    min(page_height, bbox[3]),
                )

                # 테이블 내용을 텍스트로
                table_data = table.extract()
                rows_as_text = []
                for row in table_data:
                    safe_row = [cell if cell is not None else "" for cell in row]
                    rows_as_text.append("\t".join(safe_row))
                table_text = "\n".join(rows_as_text)

                try:
                    cropped_page = plumber_page.crop(clipped_bbox)
                    table_image = cropped_page.to_image(resolution=200)

                    table_filename = f"page_{page_index+1}_table_{table_idx+1}.png"
                    table_path = self.tables_dir / table_filename
                    table_image.save(str(table_path), format="PNG")

                    elements.append(
                        {
                            "type": "table",
                            "filename": table_filename,
                            "path": str(table_path),
                            "bbox": [float(b) for b in bbox],
                            "index": table_idx + 1,
                            "text": table_text,
                            # local_text, media_text는 나중에 붙임
                        }
                    )
                except Exception as e:
                    print(
                        f"  경고: 페이지 {page_index+1}의 테이블 {table_idx+1} 이미지 추출 실패: {str(e)}"
                    )
                    print("  텍스트만 저장합니다.")
                    elements.append(
                        {
                            "type": "table",
                            "filename": None,
                            "path": None,
                            "bbox": [float(b) for b in bbox],
                            "index": table_idx + 1,
                            "text": table_text,
                        }
                    )

            page_elements[page_index + 1] = elements

        doc.close()
        pdf.close()
        return page_elements

    # =====================================================================
    # 텍스트 블록 + 섹션 정보 추출 (bbox 포함)
    # =====================================================================

    def _detect_section_info(self, text: str) -> Dict[str, Any]:
        """
        페이지 전체 텍스트에서 섹션 / 서브섹션 헤더를 간단히 감지
        """
        section_patterns = [
            r"^\d+\.\s+",        # "1. "
            r"^\d+\.\d+\s+",     # "1.1 "
            r"^\d+\.\d+\.\d+\s+",# "1.1.1 "
            r"^Chapter\s+\d+",   # "Chapter 1"
            r"^Section\s+\d+",   # "Section 1"
            r"^제\s*\d+장",      # "제1장"
            r"^제\s*\d+절",      # "제1절"
        ]

        lines = text.split("\n")
        current_section = None
        current_subsection = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if (
                        re.match(r"^\d+\.\s+", line)
                        or re.match(r"^Chapter\s+\d+", line, re.IGNORECASE)
                        or re.match(r"^제\s*\d+장", line)
                    ):
                        current_section = line
                        current_subsection = None
                    elif re.match(r"^\d+\.\d+\s+", line) or re.match(
                        r"^제\s*\d+절", line
                    ):
                        current_subsection = line
                    break

        return {
            "section": current_section,
            "subsection": current_subsection,
        }

    def extract_text_blocks(self, pdf_path: str) -> Dict[int, Dict[str, Any]]:
        """
        PDF에서 텍스트를 페이지별, 블록별로 추출 (bbox 포함)

        Returns:
            {
              page_num: {
                "blocks": [ {"text": ..., "bbox": [...]}, ... ],
                "section": ...,
                "subsection": ...,
                "full_text": ...
              },
              ...
            }
        """
        doc = fitz.open(pdf_path)
        page_texts: Dict[int, Dict[str, Any]] = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text = page.get_text()
            section_info = self._detect_section_info(full_text)

            blocks_raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, ...)

            blocks: List[Dict[str, Any]] = []
            for b in blocks_raw:
                if len(b) < 5:
                    continue
                x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
                if not txt or not txt.strip():
                    continue
                blocks.append(
                    {
                        "text": txt,
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    }
                )

            page_texts[page_num + 1] = {
                "blocks": blocks,
                "section": section_info["section"],
                "subsection": section_info["subsection"],
                "full_text": full_text,
            }

        doc.close()
        return page_texts

    # =====================================================================
    # 이미지/테이블 주변 local text 수집
    # =====================================================================

    @staticmethod
    def _rects_intersect(a: List[float], b: List[float]) -> bool:
        """두 bbox가 교차하는지 여부 (a, b: [x0,y0,x1,y1])"""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        if ax1 < bx0 or bx1 < ax0:
            return False
        if ay1 < by0 or by1 < ay0:
            return False
        return True

    def attach_local_text(
        self,
        page_elements: Dict[int, List[Dict[str, Any]]],
        page_texts: Dict[int, Dict[str, Any]],
        margin: float = 40.0,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        각 이미지/테이블에 주변 텍스트(local_text)를 붙여준다.

        전략:
        - 요소 bbox를 상하좌우로 margin 만큼 확대한 영역과
          텍스트 블록 bbox가 겹치는지 확인
        - 겹치는 블록들을 y,x 순으로 정렬해서 이어붙여 local_text로 저장
        """
        for page_num, elements in page_elements.items():
            text_info = page_texts.get(page_num, {})
            blocks = text_info.get("blocks", [])

            for el in elements:
                if el.get("type") not in ("image", "table"):
                    continue
                bbox = el.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = bbox
                expanded = [
                    x0 - margin,
                    y0 - margin,
                    x1 + margin,
                    y1 + margin,
                ]

                related_blocks: List[Tuple[float, float, str]] = []
                for blk in blocks:
                    bb = blk.get("bbox")
                    txt = blk.get("text", "")
                    if not bb or not txt.strip():
                        continue
                    if self._rects_intersect(expanded, bb):
                        by0 = bb[1]
                        bx0 = bb[0]
                        related_blocks.append((by0, bx0, txt))

                if related_blocks:
                    related_blocks.sort(key=lambda x: (x[0], x[1]))
                    texts = [re.sub(r"\s+", " ", t[2].strip()) for t in related_blocks]
                    local_text = " ".join(texts)
                    el["local_text"] = local_text
                else:
                    el["local_text"] = None

        return page_elements

    # =====================================================================
    # 텍스트 임베딩
    # =====================================================================

    def embed_text(self, text: str) -> np.ndarray:
        """
        텍스트를 임베딩 벡터로 변환
        - BGE-M3: CLS 토큰 + L2 정규화
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.embed_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            embeddings = cls_embeddings.cpu().numpy()

        return embeddings[0]

    # =====================================================================
    # 메인 처리 로직
    # =====================================================================

    def process_manual(self, pdf_path: str):
        """
        매뉴얼 PDF 파일을 처리하여 임베딩
        """
        pdf_path = str(pdf_path)
        pdf_name = Path(pdf_path).name

        print(f"\nProcessing PDF: {pdf_path}")

        # 1. 이미지와 테이블 추출
        print("Step 1: Extracting images and tables...")
        page_elements = self.extract_images_and_tables(pdf_path)

        # 2. 텍스트 블록 + 섹션 정보 추출
        print("Step 2: Extracting text blocks with bbox...")
        page_texts = self.extract_text_blocks(pdf_path)

        # 3. 이미지/테이블에 local text 부착
        print("Step 3: Collecting local text for images/tables...")
        page_elements = self.attach_local_text(page_elements, page_texts)

        # 4. 임베딩 및 저장
        print("Step 4: Creating embeddings...")
        documents: List[str] = []
        embeddings_list: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        # media 전용 임베딩 (이미지/테이블 의미 매칭용)
        media_documents: List[str] = []
        media_embeddings_list: List[List[float]] = []
        media_metadatas: List[Dict[str, Any]] = []
        media_ids: List[str] = []

        doc_id = 0
        media_id_counter = 0
        total_documents_this_file = 0

        for page_num in sorted(page_elements.keys()):
            elements = page_elements[page_num]
            text_info = page_texts.get(
                page_num,
                {"blocks": [], "section": None, "subsection": None},
            )
            blocks = text_info.get("blocks", [])
            section = text_info.get("section")
            subsection = text_info.get("subsection")

            # -------------------------------------------------------------
            # (A) 이미지/테이블에 대한 임베딩
            # -------------------------------------------------------------
            for element in elements:
                element_type = element["type"]
                element_name = element["filename"]

                # -----------------------------
                # (A-1) main 컬렉션용 metadata
                # -----------------------------
                metadata: Dict[str, Any] = {
                    "pdf_path": pdf_path,
                    "pdf_name": pdf_name,
                    "page": int(page_num),
                    "type": element_type,
                    "filename": element_name,
                    "path": element["path"],
                    "index": int(element["index"]),
                }

                # bbox는 Chroma 메타데이터에는 넣지 않는다 (list → 에러 방지)
                # 대신 metadata.json 안의 page_elements에 그대로 저장됨

                # local_text / table_text (Chroma 메타데이터에는 string OK)
                if element.get("local_text"):
                    metadata["local_text"] = element["local_text"]
                if element_type == "table" and element.get("text"):
                    metadata["table_text"] = element["text"][:2000]

                if section:
                    metadata["section"] = section
                if subsection:
                    metadata["subsection"] = subsection

                # main 컬렉션용 임베딩 텍스트 (기존과 유사)
                embed_text_parts = [f"Page {page_num}: {element_type} {element_name}"]
                if section:
                    embed_text_parts.append(f"Section: {section}")
                if subsection:
                    embed_text_parts.append(f"Subsection: {subsection}")
                if element_type == "table" and element.get("text"):
                    embed_text_parts.append(f"Content: {element['text'][:200]}")
                if element.get("local_text"):
                    embed_text_parts.append(
                        f"NearbyText: {element['local_text'][:200]}"
                    )

                embed_text_str = " | ".join(embed_text_parts)

                embedding = self.embed_text(embed_text_str)

                documents.append(embed_text_str)
                embeddings_list.append(embedding.tolist())
                metadatas.append(metadata)
                ids.append(f"doc_{doc_id}")
                doc_id += 1
                total_documents_this_file += 1

                # ----------------------------------------------
                # (A-2) media 컬렉션용 media_text + 임베딩 추가
                # ----------------------------------------------
                media_parts: List[str] = []

                # 섹션/서브섹션 정보
                if section:
                    media_parts.append(section)
                if subsection:
                    media_parts.append(subsection)

                # 주변 텍스트
                if element.get("local_text"):
                    media_parts.append(element["local_text"][:300])

                # 테이블인 경우 실제 내용 일부
                if element_type == "table" and element.get("text"):
                    media_parts.append(element["text"][:300])

                # 아무것도 없으면 fallback
                if not media_parts:
                    media_parts.append(
                        f"Page {page_num} {element_type} {element_name}"
                    )

                media_text = " ".join(media_parts)
                # 불필요한 공백 정리
                media_text = re.sub(r"\s+", " ", media_text).strip()

                # media_text를 element에도 저장하여 metadata.json에 남겨둠
                element["media_text"] = media_text

                media_embedding = self.embed_text(media_text)

                media_meta: Dict[str, Any] = {
                    "pdf_path": pdf_path,
                    "pdf_name": pdf_name,
                    "page": int(page_num),
                    "type": element_type,
                    "filename": element_name,
                    "path": element["path"],
                    "index": int(element["index"]),
                }
                if section:
                    media_meta["section"] = section
                if subsection:
                    media_meta["subsection"] = subsection

                media_documents.append(media_text)
                media_embeddings_list.append(media_embedding.tolist())
                media_metadatas.append(media_meta)
                media_ids.append(
                    f"media_{pdf_name}_{page_num}_{element_type}_{element['index']}_{media_id_counter}"
                )
                media_id_counter += 1

            # -------------------------------------------------------------
            # (B) 텍스트 블록 임베딩 (main 컬렉션용)
            # -------------------------------------------------------------
            for block_idx, block in enumerate(blocks):
                text = block["text"]

                if not text or len(text.strip()) < 5:
                    continue

                metadata: Dict[str, Any] = {
                    "pdf_path": pdf_path,
                    "pdf_name": pdf_name,
                    "page": int(page_num),
                    "type": "text",
                    "block": int(block_idx + 1),
                }
                # bbox는 마찬가지로 Chroma 메타데이터에는 넣지 않음

                if section:
                    metadata["section"] = section
                if subsection:
                    metadata["subsection"] = subsection

                embed_text_str = f"Page {page_num}: {text}"
                if section:
                    embed_text_str = f"Section {section}: {embed_text_str}"
                if subsection:
                    embed_text_str = f"Subsection {subsection}: {embed_text_str}"

                embedding = self.embed_text(embed_text_str)

                documents.append(embed_text_str)
                embeddings_list.append(embedding.tolist())
                metadatas.append(metadata)
                ids.append(f"doc_{doc_id}")
                doc_id += 1
                total_documents_this_file += 1

        # -------------------------
        # main 컬렉션 저장
        # -------------------------
        print(f"Step 5: Saving {len(documents)} embeddings to vector database...")
        if documents:
            self.collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            print("  경고: 생성된 문서가 없습니다. main 임베딩을 추가하지 않습니다.")

        # -------------------------
        # media 컬렉션 저장
        # -------------------------
        print(f"Step 5-2: Saving {len(media_documents)} media embeddings...")
        if media_documents:
            self.media_collection.add(
                embeddings=media_embeddings_list,
                documents=media_documents,
                metadatas=media_metadatas,
                ids=media_ids,
            )
        else:
            print("  경고: 생성된 media 임베딩이 없습니다.")

        # 6. metadata.json 업데이트 (여러 PDF 누적)
        print("Step 6: Updating metadata.json...")
        all_meta = self._load_metadata_all()

        # 기존에 같은 pdf_path/pdf_name이 있으면 제거 후 다시 추가
        files = all_meta.get("files", [])
        new_files = []
        for f in files:
            if f.get("pdf_path") == pdf_path or f.get("pdf_name") == pdf_name:
                continue
            new_files.append(f)

        new_files.append(
            {
                "pdf_path": pdf_path,
                "pdf_name": pdf_name,
                "total_pages": len(page_elements),
                "total_documents": total_documents_this_file,
                # page_elements 안에는 local_text, media_text 등이 포함되어 있음
                "page_elements": {str(k): v for k, v in page_elements.items()},
                "page_texts": {
                    str(k): {
                        "section": page_texts[k].get("section"),
                        "subsection": page_texts[k].get("subsection"),
                        "full_text": page_texts[k].get("full_text"),
                        "blocks": page_texts[k].get("blocks"),
                    }
                    for k in page_texts.keys()
                },
            }
        )

        all_meta["files"] = new_files
        self._save_metadata_all(all_meta)

        print(
            f"Processing complete! {total_documents_this_file} documents embedded for {pdf_name}."
        )
        print(f"Output directory: {self.output_dir}")

    # =====================================================================
    # 간단 검색 함수 (테스트용)
    # =====================================================================

    def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 검색 (벡터 기반, main 컬렉션)
        """
        query_embedding = self.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )

        formatted_results: List[Dict[str, Any]] = []
        if results.get("documents") and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "id": results["ids"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )

        return formatted_results

    def search_media(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        미디어 컬렉션에 대한 검색 (이미지/테이블 의미 매칭용).
        나중에 manualllm에서 답변 단락 텍스트로 여기를 직접 호출해서
        가장 관련 있는 이미지/테이블을 고를 때 사용 가능.
        """
        query_embedding = self.embed_text(query)

        results = self.media_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )

        formatted_results: List[Dict[str, Any]] = []
        if results.get("documents") and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "document": results["documents"][0][i],  # media_text
                        "metadata": results["metadatas"][0][i],  # pdf_name/page/type/filename 등
                        "id": results["ids"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )

        return formatted_results


def main():
    """PDF 업로드 폴더의 모든 PDF 파일을 임베딩"""
    upload_dir = Path("pdf_uploads")
    upload_dir.mkdir(exist_ok=True)

    pdf_files = list(upload_dir.glob("*.pdf"))

    if not pdf_files:
        print(
            f"PDF 파일을 찾을 수 없습니다. '{upload_dir}' 폴더에 PDF 파일을 업로드해주세요."
        )
        return

    print(f"총 {len(pdf_files)}개의 PDF 파일을 찾았습니다.")

    embedder = ManualEmbedder(output_dir="manual_output")

    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"처리 중: {pdf_path.name}")
        print(f"{'='*60}")
        try:
            embedder.process_manual(str(pdf_path))
            print(f"✓ {pdf_path.name} 처리 완료")
        except Exception as e:
            print(f"✗ {pdf_path.name} 처리 중 오류 발생: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("모든 PDF 파일 처리 완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
