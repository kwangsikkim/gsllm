"""
하이브리드 검색 기반 질문-답변 시스템
- 텍스트 기반 검색 (BM25) 50개 + 벡터 검색 (RAG) 50개
- 리랭커로 상위 10개 선택
- LLM으로 답변 생성
- 텍스트가 등장한 페이지의 관련 이미지/테이블 선택적 포함
- media_text 임베딩 기반으로 의미적으로 가까운 이미지/테이블 우선 선택
- 답변 중간에 이미지/테이블 삽입
"""

# ========================================================================
# 질문 작성 영역 - 여기에 질문을 작성하세요
# ========================================================================
QUESTION = """
만화의 창작 및 제작지원에 대해 알려줘
"""
# ========================================================================

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import re

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import chromadb
from rank_bm25 import BM25Okapi
from llama_cpp import Llama


# 모델 경로 설정
LLM_MODEL_PATH = "/home/siwasoft/gpt-oss-20b-GGUF/gpt-oss-20b-Q2_K_L.gguf"
EMBED_MODEL = "BAAI/bge-m3"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
OUTPUT_DIR = "/home/siwasoft/gsllm/manual_output"


# =========================
# 한국어용 간단 토크나이저
# =========================
def tokenize(text: str) -> List[str]:
    """
    BM25용 간단 토크나이저 (문자 bi-gram 기반)
    - 기호 제거
    - 공백 제거
    - 연속된 2글자씩 잘라서 토큰으로 사용
    """
    text = text.lower()
    # 한글/영문/숫자/공백만 남기기
    text = re.sub(r"[^0-9a-z가-힣 ]", " ", text)
    # 공백 제거
    text = text.replace(" ", "")
    if len(text) <= 1:
        return [text] if text else []
    return [text[i:i+2] for i in range(len(text) - 1)]


class HybridQASystem:
    """하이브리드 검색 기반 질문-답변 시스템"""

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        embed_model: str = EMBED_MODEL,
        reranker_name: str = RERANKER_NAME,
        llm_model_path: str = LLM_MODEL_PATH,
    ):
        """
        Args:
            output_dir: 임베딩된 데이터가 저장된 디렉토리
            embed_model: 임베딩 모델 이름
            reranker_name: 리랭커 모델 이름
            llm_model_path: LLM 모델 경로 (GGUF)
        """
        self.output_dir = Path(output_dir)

        # GPU 사용 가능 여부 확인
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading embedding model...")
        # 임베딩 모델 로드
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.embed_model = AutoModel.from_pretrained(embed_model)
        self.embed_model = self.embed_model.to(self.device)
        self.embed_model.eval()

        print("Loading reranker model...")
        # 리랭커 모델 로드 (BGE reranker는 AutoModelForSequenceClassification 사용)
        from transformers import AutoModelForSequenceClassification

        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_name
        )
        self.reranker_model = self.reranker_model.to(self.device)
        self.reranker_model.eval()

        print("Loading LLM model...")
        # LLM 모델 로드 (GGUF)
        n_gpu_layers = 0  # CPU만 사용
        if torch.cuda.is_available():
            n_gpu_layers = -1  # -1은 모든 레이어를 GPU로 이동
            print(f"  GPU 사용: {n_gpu_layers} layers on GPU")
        else:
            print("  CPU 사용")

        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=8192,  # 컨텍스트 길이
            n_threads=4,  # 스레드 수
            n_gpu_layers=n_gpu_layers,  # GPU 레이어 수
            verbose=False,
        )

        # ChromaDB 연결
        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=str(self.output_dir / "chroma_db")
        )
        self.collection = self.client.get_collection("manual_embeddings")

        # media_text용 컬렉션 (manualemb 쪽에서 만들어 둔 것)
        self.media_collection = None
        try:
            self.media_collection = self.client.get_collection(
                "manual_media_embeddings"
            )
            print("Connected to media collection: manual_media_embeddings")
        except Exception as e:
            print(f"  경고: media 컬렉션(manual_media_embeddings)을 찾지 못했습니다: {e}")

        # BM25를 위한 문서 로드
        print("Loading documents for BM25...")
        self._load_documents_for_bm25()

        # 메타데이터 로드
        print("Loading metadata...")
        self.metadata = self._load_metadata()

        # 여러 PDF를 지원하는 페이지 요소 인덱스 구축
        self._build_metadata_index()

        print("System ready!")

    # ------------------------------------------------------------------
    # 메타데이터 로딩 및 인덱싱
    # ------------------------------------------------------------------
    def _load_metadata(self) -> Dict[str, Any]:
        """메타데이터 파일 로드"""
        metadata_path = self.output_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _build_metadata_index(self) -> None:
        """
        여러 PDF를 지원하는 페이지 요소 인덱스 생성.
        """
        self.page_elements_index: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        self.default_pdf_name: Optional[str] = None

        if not self.metadata:
            return

        files = self.metadata.get("files")
        if isinstance(files, list):
            # 새 구조: 여러 파일
            for file_entry in files:
                pdf_name = file_entry.get("pdf_name")
                pdf_path = file_entry.get("pdf_path")
                if not pdf_name and pdf_path:
                    pdf_name = os.path.basename(pdf_path)
                if not pdf_name:
                    continue

                if self.default_pdf_name is None:
                    self.default_pdf_name = pdf_name

                page_elements = file_entry.get("page_elements", {})
                for page_str, elements in page_elements.items():
                    try:
                        page_num = int(page_str)
                    except ValueError:
                        continue
                    self.page_elements_index[(pdf_name, page_num)] = elements or []
        else:
            # 예전 단일 파일 구조
            pdf_name = self.metadata.get("pdf_name")
            pdf_path = self.metadata.get("pdf_path")
            if not pdf_name and pdf_path:
                pdf_name = os.path.basename(pdf_path)
            if not pdf_name:
                pdf_name = "__single__"

            self.default_pdf_name = pdf_name
            page_elements = self.metadata.get("page_elements", {})
            for page_str, elements in page_elements.items():
                try:
                    page_num = int(page_str)
                except ValueError:
                    continue
                self.page_elements_index[(pdf_name, page_num)] = elements or []

    def _guess_pdf_name_for_page(self, page_num: int) -> Optional[str]:
        """pdf_name 정보를 못 찾았을 때, page 번호만으로 추정 (후방 호환용)."""
        candidates: List[str] = []
        for pdf_name, p in self.page_elements_index.keys():
            if p == page_num:
                candidates.append(pdf_name)

        if len(candidates) == 1:
            return candidates[0]
        return self.default_pdf_name

    def _get_page_elements(
        self, pdf_name: Optional[str], page_num: int
    ) -> List[Dict[str, Any]]:
        """(pdf_name, page_num)에 해당하는 페이지 요소 리스트 반환."""
        if not self.page_elements_index:
            return []

        key = (pdf_name or self.default_pdf_name, page_num)
        if key in self.page_elements_index:
            return self.page_elements_index[key]

        guessed = self._guess_pdf_name_for_page(page_num)
        if guessed:
            return self.page_elements_index.get((guessed, page_num), [])
        return []

    # ------------------------------------------------------------------
    # 검색 관련
    # ------------------------------------------------------------------
    def _load_documents_for_bm25(self):
        """BM25 검색을 위해 모든 문서 로드 (빈 컬렉션 방어 추가)"""
        # ChromaDB에서 모든 문서 가져오기
        all_data = self.collection.get()

        self.documents = all_data.get("documents", []) or []
        self.metadatas = all_data.get("metadatas", []) or []
        self.ids = all_data.get("ids", []) or []

        if not self.documents:
            print("  경고: 벡터 DB에 문서가 없어 BM25 인덱스를 생성하지 않습니다.")
            self.bm25 = None
            return

        # BM25를 위한 토큰화된 문서 (한국어 대응: 문자 bi-gram 토큰 사용)
        tokenized_docs = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def embed_text(self, text: str) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환 (BGE-M3 권장 방식: CLS + L2 정규화)"""
        with torch.no_grad():
            inputs = self.embed_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            # GPU로 입력 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.embed_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            embeddings = cls_embeddings.cpu().numpy()
        return embeddings[0]

    def text_search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """BM25 기반 텍스트 검색"""
        if getattr(self, "bm25", None) is None:
            return []

        # 한국어 대응 토크나이저 사용
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 상위 k개 선택
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            # 점수가 0이어도 상대적 순위는 의미 있으니 그대로 사용
            results.append(
                {
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "id": self.ids[idx],
                    "score": float(scores[idx]),
                    "search_type": "text",
                }
            )

        return results

    def vector_search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """벡터 기반 검색 (RAG)"""
        query_embedding = self.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        formatted_results: List[Dict[str, Any]] = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "id": results["ids"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                        "search_type": "vector",
                    }
                )

        return formatted_results

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """리랭커를 사용하여 후보 문서 재정렬"""
        if not candidates:
            return []

        # 중복 제거 (id 기준)
        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate["id"] not in seen_ids:
                unique_candidates.append(candidate)
                seen_ids.add(candidate["id"])

        if len(unique_candidates) <= top_k:
            return unique_candidates

        # 리랭커를 위한 입력 준비
        pairs = [(query, candidate["document"]) for candidate in unique_candidates]

        with torch.no_grad():
            scores: List[float] = []
            batch_size = 16

            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                batch_queries = [pair[0] for pair in batch_pairs]
                batch_docs = [pair[1] for pair in batch_pairs]

                # 토큰화
                inputs = self.reranker_tokenizer(
                    batch_queries,
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # GPU로 입력 이동
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.reranker_model(**inputs)

                # BGE reranker는 logits를 반환 (AutoModelForSequenceClassification 사용 시)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores.tolist())

        # 점수와 함께 정렬
        scored_candidates = [
            {**candidate, "rerank_score": score}
            for candidate, score in zip(unique_candidates, scores)
        ]

        # 점수 기준으로 정렬 (내림차순)
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_candidates[:top_k]

    def hybrid_search(
        self, query: str, text_k: int = 50, vector_k: int = 50, final_k: int = 10
    ) -> List[Dict[str, Any]]:
        """하이브리드 검색: 텍스트 검색 + 벡터 검색 후 리랭킹"""
        print(f"\n검색 중: '{query}'")

        # 1. 텍스트 기반 검색
        print(f"텍스트 검색 (BM25): 상위 {text_k}개...")
        text_results = self.text_search(query, top_k=text_k)
        print(f"  → {len(text_results)}개 발견")

        # 2. 벡터 기반 검색
        print(f"벡터 검색 (RAG): 상위 {vector_k}개...")
        vector_results = self.vector_search(query, top_k=vector_k)
        print(f"  → {len(vector_results)}개 발견")

        # 3. 결과 병합
        all_candidates = text_results + vector_results
        print(f"총 {len(all_candidates)}개 후보 문서")

        # 4. 리랭킹
        print(f"리랭킹 중: 상위 {final_k}개 선택...")
        reranked_results = self.rerank(query, all_candidates, top_k=final_k)
        print(f"  → 최종 {len(reranked_results)}개 선택됨")

        return reranked_results

    # ------------------------------------------------------------------
    # media 컬렉션 검색 (media_text 기반)
    # ------------------------------------------------------------------
    def media_search(self, text: str, top_k: int = 80) -> List[Dict[str, Any]]:
        """
        manual_media_embeddings 컬렉션에 대해 벡터 검색
        (각 media는 media_text를 임베딩한 것)
        """
        if self.media_collection is None:
            return []

        q_emb = self.embed_text(text)
        results = self.media_collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
        )

        formatted: List[Dict[str, Any]] = []
        if results.get("documents") and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                formatted.append(
                    {
                        "document": results["documents"][0][i],  # media_text 요약
                        "metadata": results["metadatas"][0][i],
                        "id": results["ids"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )
        return formatted

    # ------------------------------------------------------------------
    # 이미지 / 테이블 추출 (의미 기반 + 페이지 필터)
    # ------------------------------------------------------------------
    def extract_images_and_tables(
        self, query: str, results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        검색 결과에서 이미지와 테이블 경로 추출.

        - 리랭커 상위 결과들 중에서 type == "text" 인 것의 (pdf_name, page)를 모은다.
        - manual_media_embeddings에서 media_text 기준으로 query와 가장
          비슷한 media를 가져와서, 위 (pdf_name, page) 안에 있는 것만 남긴다.
        - media 컬렉션이 없으면, 예전 방식(page_elements 기반)으로 fallback.
        """
        related_images: List[Dict[str, Any]] = []
        related_tables: List[Dict[str, Any]] = []
        seen_paths: Set[str] = set()

        # 1) 텍스트가 등장한 페이지 수집 (연속된 페이지도 포함)
        text_pages: Set[Tuple[Optional[str], int]] = set()
        pages_by_pdf: Dict[Optional[str], Set[int]] = {}
        
        for result in results:
            metadata = result.get("metadata", {})
            if metadata.get("type") == "text":
                page = metadata.get("page")
                if page is None:
                    continue
                pdf_name = metadata.get("pdf_name")
                text_pages.add((pdf_name, page))
                
                # PDF별로 페이지 수집
                if pdf_name not in pages_by_pdf:
                    pages_by_pdf[pdf_name] = set()
                pages_by_pdf[pdf_name].add(page)
        
        # 각 PDF의 페이지에 대해 연속된 페이지 추가 (±1, ±2)
        # 이렇게 하면 연속된 페이지의 미디어(이미지/테이블)도 포함됨
        for pdf_name, pages in pages_by_pdf.items():
            for page in pages:
                if isinstance(page, int):
                    for offset in [-2, -1, 1, 2]:
                        adjacent_page = page + offset
                        if adjacent_page > 0:
                            text_pages.add((pdf_name, adjacent_page))

        # media 컬렉션이 있을 때: 의미 기반 우선
        if self.media_collection is not None and text_pages:
            media_results = self.media_search(query, top_k=80)

            for mr in media_results:
                md = mr.get("metadata", {})
                mtype = md.get("type")
                pdf_name = md.get("pdf_name")
                page = md.get("page")

                if (pdf_name, page) not in text_pages:
                    continue

                path = md.get("path")
                filename = md.get("filename")
                if not path or not filename:
                    continue
                if not os.path.exists(path):
                    continue
                if path in seen_paths:
                    continue

                # sidebar 류는 manualemb에서 이미 안 집어넣었겠지만, 방어적으로 한 번 더
                if mtype == "image" and md.get("is_sidebar", False):
                    continue

                summary = mr.get("document", "")
                summary = re.sub(r"\s+", " ", summary).strip()

                media_info = {
                    "path": path,
                    "filename": filename,
                    "pdf_name": pdf_name,
                    "page": page,
                    "bbox": md.get("bbox"),
                    "summary": summary[:160] if summary else None,
                    "type": mtype,
                }

                if mtype == "image":
                    related_images.append(media_info)
                elif mtype == "table":
                    related_tables.append(media_info)

                seen_paths.add(path)

        # fallback: media 컬렉션이 없거나 text_pages가 비어있을 때
        if not related_images and not related_tables:
            # 2) 검색 결과에 직접 등장한 이미지/테이블 추가 (기존 로직 유지)
            for result in results:
                metadata = result.get("metadata", {})
                mtype = metadata.get("type")
                if mtype not in ("image", "table"):
                    continue

                # B안: 검색 결과에서 직접 나온 이미지가 사이드바면 제외
                if mtype == "image" and metadata.get("is_sidebar", False):
                    continue

                path = metadata.get("path")
                if not path or not os.path.exists(path):
                    continue

                filename = metadata.get("filename", "")
                pdf_name = metadata.get("pdf_name")
                page = metadata.get("page")
                media_info = {
                    "path": path,
                    "filename": filename,
                    "pdf_name": pdf_name,
                    "page": page,
                    "bbox": metadata.get("bbox"),
                    "summary": None,
                    "type": mtype,
                }

                if mtype == "image":
                    if path not in seen_paths:
                        related_images.append(media_info)
                        seen_paths.add(path)
                else:
                    if path not in seen_paths:
                        related_tables.append(media_info)
                        seen_paths.add(path)

            # 3) 텍스트가 있는 페이지에서 연관 이미지/테이블 추가 (기존 로직)
            for pdf_name, page in text_pages:
                if page is None:
                    continue

                elements = self._get_page_elements(pdf_name, page)
                for el in elements:
                    etype = el.get("type")
                    path = el.get("path")
                    if not path or not os.path.exists(path):
                        continue
                    if path in seen_paths:
                        continue

                    # B안: page_elements에서도 사이드바 이미지 제외
                    if etype == "image" and el.get("is_sidebar", False):
                        continue

                    filename = el.get("filename", "")
                    media_info = {
                        "path": path,
                        "filename": filename,
                        "pdf_name": pdf_name,
                        "page": page,
                        "bbox": el.get("bbox"),
                        "summary": None,
                        "type": etype,
                    }

                    if etype == "image":
                        related_images.append(media_info)
                    elif etype == "table":
                        related_tables.append(media_info)
                    seen_paths.add(path)

        # 너무 많으면 앞쪽 몇 개만 사용 (LLM 프롬프트 길이 방지)
        related_images = related_images[:10]
        related_tables = related_tables[:10]

        return {"images": related_images, "tables": related_tables}

    # ------------------------------------------------------------------
    # 컨텍스트 포맷팅
    # ------------------------------------------------------------------
    def format_context(
        self, results: List[Dict[str, Any]], max_doc_length: int = 500
    ) -> str:
        """검색 결과를 LLM 입력용 컨텍스트로 포맷팅"""
        # 페이지 번호 순으로 정렬하여 전후 관계를 명확히 함
        results_with_page = []
        for result in results:
            metadata = result.get("metadata", {})
            page = metadata.get("page")
            if isinstance(page, int):
                results_with_page.append((page, result))
            else:
                results_with_page.append((999999, result))  # 페이지 없는 것은 뒤로
        
        # 페이지 번호 순으로 정렬
        results_with_page.sort(key=lambda x: x[0])
        
        context_parts: List[str] = []

        # 페이지별 이미지/테이블 정보 수집 (파일명 기준)
        page_media: Dict[int, Dict[str, List[str]]] = {}
        for _, result in results_with_page:
            metadata = result.get("metadata", {})
            page = metadata.get("page")
            if page:
                if page not in page_media:
                    page_media[page] = {"images": [], "tables": []}
                element_type = metadata.get("type", "")
                filename = metadata.get("filename", "")
                if element_type == "image" and filename:
                    # 사이드바는 굳이 안 넣어도 됨
                    if metadata.get("is_sidebar", False):
                        continue
                    page_media[page]["images"].append(filename)
                elif element_type == "table" and filename:
                    page_media[page]["tables"].append(filename)

        for i, (page, result) in enumerate(results_with_page, 1):
            doc = result.get("document", "")
            # 문서 길이 제한
            if len(doc) > max_doc_length:
                doc = doc[:max_doc_length] + "..."

            metadata = result.get("metadata", {})
            page_num = page if isinstance(page, int) else "?"
            element_type = metadata.get("type", "text")

            context_str = f"[문서 {i}] (페이지 {page_num}, 유형: {element_type})\n{doc}"

            # 해당 페이지의 이미지/테이블 정보 추가
            if isinstance(page, int) and page in page_media:
                media_info = []
                if page_media[page]["images"]:
                    media_info.append(
                        f"이미지: {', '.join(page_media[page]['images'])}"
                    )
                if page_media[page]["tables"]:
                    media_info.append(
                        f"테이블: {', '.join(page_media[page]['tables'])}"
                    )
                if media_info:
                    context_str += f"\n[참고 자료: {', '.join(media_info)}]"

            context_parts.append(context_str + "\n")

        return "\n".join(context_parts)

    # ------------------------------------------------------------------
    # 답변 생성 후 단락별 의미 기반 media 삽입
    # ------------------------------------------------------------------
    def _attach_semantic_media_to_answer(
        self, answer: str, images: List[Dict[str, Any]], tables: List[Dict[str, Any]]
    ) -> str:
        """
        - 최종 답변을 단락(빈 줄 기준)으로 나눈 뒤,
        - 각 단락 내용과 media_text(= media_collection)를 의미적으로 비교해서
          가장 가까운 이미지/테이블을 그 단락 바로 아래에 붙인다.
        - 이미 단락 안에 [이미지:], [테이블:]이 있으면 추가하지 않음.
        """
        if not (images or tables):
            return answer
        if self.media_collection is None:
            return answer

        # filename -> media_info 맵 (후보군)
        candidate_by_filename: Dict[str, Dict[str, Any]] = {}
        for m in images + tables:
            fn = m.get("filename")
            if not fn:
                continue
            candidate_by_filename[fn] = m

        if not candidate_by_filename:
            return answer

        used_filenames: Set[str] = set()

        # 단락 분할 (두 줄 이상 공백 기준이 아니라, 간단하게 빈 줄 기준)
        segments = answer.split("\n\n")
        new_segments: List[str] = []

        for seg in segments:
            seg_stripped = seg.strip()
            new_segments.append(seg)

            # 너무 짧은 단락이거나, 이미 media 참조가 있으면 스킵
            if len(seg_stripped) < 40:
                continue
            if "[이미지:" in seg or "[테이블:" in seg or "위와 관련된 자료" in seg:
                continue

            # 이 단락 내용으로 media_search 수행
            media_results = self.media_search(seg_stripped, top_k=30)
            chosen_tokens: List[str] = []

            for mr in media_results:
                md = mr.get("metadata", {})
                fn = md.get("filename")
                if not fn or fn not in candidate_by_filename:
                    continue
                if fn in used_filenames:
                    continue

                mtype = md.get("type")
                if mtype == "image":
                    prefix = "이미지"
                elif mtype == "table":
                    prefix = "테이블"
                else:
                    continue

                chosen_tokens.append(f"[{prefix}: {fn}]")
                used_filenames.add(fn)

                # 단락당 최대 2개까지만 붙이도록
                if len(chosen_tokens) >= 2:
                    break

            if chosen_tokens:
                # 단락 바로 아래에 한 줄 추가
                new_segments.append(
                    "(위와 관련된 자료: " + " ".join(chosen_tokens) + ")"
                )

        return "\n\n".join(new_segments)

    # ------------------------------------------------------------------
    # 답변 정제
    # ------------------------------------------------------------------
    def _clean_answer(self, answer: str) -> str:
        """
        LLM이 생성한 답변에서 불필요한 부분 제거
        - 생각 과정 부분 제거 (<|end|><|start|>assistant 이전)
        - 참고 문서 정보 제거 (참고 문서:, 이미지:, 테이블: 등)
        """
        # 1. <|end|><|start|>assistant<|channel|>final<|message|> 이후 부분만 추출
        marker = "<|end|><|start|>assistant<|channel|>final<|message|>"
        if marker in answer:
            answer = answer.split(marker, 1)[1].strip()
        
        # 2. "(위와 관련된 자료: ...)" 패턴은 제거하지 않음 (테이블/이미지 참조를 위해 유지)
        # answer = re.sub(r'\(위와 관련된 자료:\s*[^)]+\)', '', answer)
        
        # 3. "참고 문서:", "참고 문서: 10개" 같은 참고 정보 제거
        # "참고 문서:" 또는 "참고 문서: "로 시작하는 줄부터 끝까지 제거
        lines = answer.split('\n')
        cleaned_lines = []
        skip_rest = False
        
        for line in lines:
            # 참고 문서 섹션 시작 감지
            if line.strip().startswith("참고 문서:") or line.strip().startswith("참고 문서"):
                skip_rest = True
                break
            # 구분선 이후 제거
            if line.strip().startswith("---") and len(line.strip()) > 10:
                # 이전에 이미 내용이 있으면 여기서 멈춤
                if cleaned_lines:
                    break
            # "이미지", "테이블"로 시작하는 참고 정보 줄 제거 (단, [이미지:], [테이블:] 형식은 유지)
            if line.strip().startswith("이미지 ") or line.strip().startswith("테이블 "):
                # [이미지:], [테이블:] 형식이 포함된 줄은 유지
                if "[이미지:" not in line and "[테이블:" not in line:
                    if ":" in line or "-" in line:
                        skip_rest = True
                        break
            cleaned_lines.append(line)
        
        answer = '\n'.join(cleaned_lines).strip()
        
        # 4. "------------------------------------------------------------" 같은 구분선 제거
        answer = re.sub(r'^[-=]{20,}.*$', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'\n[-=]{20,}.*$', '', answer, flags=re.MULTILINE)
        
        # 5. 연속된 빈 줄 정리
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        return answer.strip()

    # ------------------------------------------------------------------
    # 답변 생성 및 미디어 삽입 (기존 형식 유지 + 의미 기반 보강)
    # ------------------------------------------------------------------
    def _insert_media_in_answer(
        self, answer: str, images: List[Dict[str, Any]], tables: List[Dict[str, Any]]
    ) -> str:
        """
        답변 중간에 이미지/테이블 삽입.

        - LLM이 특수 마커([IMAGE:top] 등)를 사용했다면 실제 파일명으로 교체
        - 그렇지 않으면, 적당한 위치(페이지 언급 후 등)에 [이미지: 파일명] / [테이블: 파일명]을 삽입
        """
        # 이미지와 테이블을 위치별로 그룹화 (position 없으면 middle로 취급)
        media_by_position = {
            "top": {"images": [], "tables": []},
            "middle": {"images": [], "tables": []},
            "bottom": {"images": [], "tables": []},
        }

        for img in images:
            pos = img.get("position", "middle")
            media_by_position[pos]["images"].append(img)

        for table in tables:
            pos = table.get("position", "middle")
            media_by_position[pos]["tables"].append(table)

        result = answer

        # 1) LLM이 사용한 마커 치환: [IMAGE:top], [TABLE:middle] 등
        for position in ["top", "middle", "bottom"]:
            # 이미지 마커 교체
            img_marker = f"[IMAGE:{position}]"
            if img_marker in result:
                img_refs = []
                for img in media_by_position[position]["images"]:
                    filename = img.get("filename", "")
                    if filename:
                        img_refs.append(f"[이미지: {filename}]")
                replacement = "\n".join(img_refs) if img_refs else ""
                result = result.replace(img_marker, replacement)

            # 테이블 마커 교체
            table_marker = f"[TABLE:{position}]"
            if table_marker in result:
                table_refs = []
                for table in media_by_position[position]["tables"]:
                    filename = table.get("filename", "")
                    if filename:
                        table_refs.append(f"[테이블: {filename}]")
                replacement = "\n".join(table_refs) if table_refs else ""
                result = result.replace(table_marker, replacement)

        # 2) 마커도 없고 직접 참조([이미지:, 테이블:])도 없으면, 자동 삽입 시도
        if "[이미지:" not in result and "[테이블:" not in result:
            lines = result.split("\n")
            new_lines = []

            # 모든 미디어를 파일명으로 매핑
            all_media: Dict[str, str] = {}
            for img in images:
                filename = img.get("filename", "")
                if filename:
                    all_media[filename] = "이미지"
            for table in tables:
                filename = table.get("filename", "")
                if filename:
                    all_media[filename] = "테이블"

            for i, line in enumerate(lines):
                new_lines.append(line)

                # "페이지 N" 같은 언급이 있으면, 해당 페이지의 미디어를 붙이기 시도
                page_matches = re.findall(r"페이지\s*(\d+)", line)
                if page_matches:
                    for page_str in page_matches:
                        try:
                            page_num = int(page_str)
                            page_media_tokens: List[str] = []
                            for filename, media_type in all_media.items():
                                if f"page_{page_num}_" in filename:
                                    if media_type == "이미지":
                                        page_media_tokens.append(
                                            f"[이미지: {filename}]"
                                        )
                                    else:
                                        page_media_tokens.append(
                                            f"[테이블: {filename}]"
                                        )
                            if (
                                page_media_tokens
                                and i == len(lines) - 1
                                or (
                                    i < len(lines) - 1
                                    and not lines[i + 1].strip().startswith("[")
                                )
                            ):
                                new_lines.append("  " + " ".join(page_media_tokens))
                        except Exception:
                            pass

            result = "\n".join(new_lines)

            # 여전히 미디어 참조가 없으면 맨 끝에 참고 자료로 추가
            if "[이미지:" not in result and "[테이블:" not in result:
                media_refs = []
                for img in images:
                    filename = img.get("filename", "")
                    if filename:
                        media_refs.append(f"[이미지: {filename}]")
                for table in tables:
                    filename = table.get("filename", "")
                    if filename:
                        media_refs.append(f"[테이블: {filename}]")
                if media_refs:
                    result += "\n\n참고 자료:\n" + "\n".join(media_refs)

        return result

    def generate_answer(
        self,
        query: str,
        context: str,
        images: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
    ) -> str:
        """LLM을 사용하여 답변 생성"""

        # 프롬프트 구성
        prompt = f"""기술 문서를 분석하여 질문에 답변하세요. 한글로 작성하세요.

중요: 참고 문서의 페이지 번호를 확인하고, 연속된 페이지의 내용도 함께 고려하여 전후 맥락을 면밀히 분석하세요. 
특히 한 주제가 여러 페이지에 걸쳐 설명되는 경우, 모든 관련 페이지의 내용을 종합하여 답변하세요.

참고 문서:
{context}

"""

        # 이미지/테이블 사용 지침 + 의미 요약 추가
        if images or tables:
            available_media = []
            for img in images:
                filename = img.get("filename", "")
                if not filename:
                    continue
                desc = img.get("summary")
                if desc:
                    available_media.append(f"이미지: {filename} (내용 요약: {desc})")
                else:
                    available_media.append(f"이미지: {filename}")
            for table in tables:
                filename = table.get("filename", "")
                if not filename:
                    continue
                desc = table.get("summary")
                if desc:
                    available_media.append(f"테이블: {filename} (내용 요약: {desc})")
                else:
                    available_media.append(f"테이블: {filename}")

            prompt += "=== 매우 중요: 이미지/테이블 참조 방법 ===\n"
            if available_media:
                # 처음 10개만 노출
                prompt += (
                    "사용 가능한 참고 자료 (파일명과 대략적인 내용):\n"
                    + "\n".join(f"- {m}" for m in available_media[:10])
                    + "\n"
                )
            else:
                prompt += "사용 가능한 참고 자료: 없음\n"
            prompt += "\n답변 작성 시 반드시 다음 규칙을 따르세요:\n"
            prompt += "1. 각 내용을 설명할 때, 해당 내용과 관련된 이미지나 테이블을 바로 그 자리에 삽입하세요.\n"
            prompt += '2. 삽입 형식: [이미지: 파일명] 또는 [테이블: 파일명]\n'
            prompt += (
                "3. 예시: '관성력에 의한 미끄러짐은 다음과 같은 현상이 발생합니다. "
                "[이미지: page_12_img_1.png] 위 그림에서 보듯이...'\n"
            )
            prompt += (
                "4. 절대 답변 끝에 참고 자료를 모아서 넣지 마세요. "
                "각 설명 중간에 해당 자료를 삽입하세요.\n"
            )
            prompt += (
                "5. 여러 자료가 관련되어 있다면 모두 나열하세요: "
                "[이미지: file1.png] [테이블: file2.png]\n\n"
            )

        prompt += f"질문: {query}\n\n답변:"

        # LLM으로 답변 생성
        response = self.llm(
            prompt,
            max_tokens=4096,  # 답변 토큰 제한
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.15,
            stop=["질문:", "[문서", "참고 문서:"],
        )

        answer = response["choices"][0]["text"].strip()
        
        # 답변 정제: 불필요한 부분 제거
        answer = self._clean_answer(answer)

        # 답변이 잘렸는지 판단
        is_truncated = False
        finish_reason = response["choices"][0].get("finish_reason", "")

        if finish_reason == "length":
            is_truncated = True

        if answer and not answer[-1] in [
            ".",
            "!",
            "?",
            "다",
            "요",
            "니다",
            "습니다",
            "니다.",
            "습니다.",
            "요.",
            "다.",
        ]:
            if len(answer) > 100:
                is_truncated = True

        if len(answer) < 20:
            answer = (
                "참고 문서를 기반으로 답변을 생성했지만 내용이 충분하지 않습니다. "
                "더 많은 컨텍스트가 필요할 수 있습니다."
            )
        elif is_truncated:
            answer += (
                "\n\n[참고: 답변이 토큰 제한으로 인해 잘렸을 수 있습니다. "
                "더 자세한 정보가 필요하면 질문을 더 구체적으로 해주세요.]"
            )

        # 1차: 마커 기반/페이지 기반 삽입
        answer = self._insert_media_in_answer(answer, images, tables)

        # 2차: 단락별 의미 기반 보강
        answer = self._attach_semantic_media_to_answer(answer, images, tables)

        return answer

    # ------------------------------------------------------------------
    # 메인 ask 엔트리
    # ------------------------------------------------------------------
    def ask(self, query: str) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        results = self.hybrid_search(query, text_k=50, vector_k=50, final_k=10)

        if not results:
            return {
                "query": query,
                "answer": "관련 문서를 찾을 수 없습니다.",
                "sources": [],
                "images": [],  # 파일 경로 목록
                "tables": [],
            }

        # 의미 기반 media + 페이지 필터 적용
        media = self.extract_images_and_tables(query, results)
        context = self.format_context(results)

        print("\n답변 생성 중...")
        answer = self.generate_answer(
            query, context, media["images"], media["tables"]
        )

        sources = []
        for result in results:
            metadata = result.get("metadata", {})
            sources.append(
                {
                    "page": metadata.get("page", "?"),
                    "type": metadata.get("type", "text"),
                    "section": metadata.get("section", ""),
                    "pdf_name": metadata.get("pdf_name", ""),
                    "rerank_score": result.get("rerank_score", 0),
                }
            )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "images": [img["path"] for img in media["images"]],
            "tables": [table["path"] for table in media["tables"]],
            "num_sources": len(results),
        }


def main():
    """질문-답변 시스템 실행"""
    qa_system = HybridQASystem()

    query = QUESTION.strip()

    default_texts = [
        "여기에 질문을 작성하세요.",
        "여기에 질문을 작성하세요.\n예: 사용 방법은 무엇인가요?",
        "",
    ]

    if not query or query in default_texts:
        print("오류: QUESTION 변수에 질문을 작성해주세요.")
        print('파일 상단의 QUESTION = """ ... """ 부분에 질문을 입력하세요.')
        return

    print("\n" + "=" * 60)
    print("질문-답변 시스템")
    print("=" * 60)
    print(f"\n질문: {query}\n")

    try:
        result = qa_system.ask(query)

        print("\n" + "=" * 60)
        print("답변:")
        print("=" * 60)
        answer_text = result["answer"]

        if "[참고: 답변이 토큰 제한으로 인해 잘렸을 수 있습니다" in answer_text:
            print(answer_text)
            print("\n⚠️  경고: 답변이 토큰 제한으로 인해 잘렸을 수 있습니다.")
        else:
            print(answer_text)

        print("\n" + "-" * 60)
        print(f"참고 문서: {result['num_sources']}개")
        for i, source in enumerate(result["sources"][:5], 1):
            print(
                f"  {i}. 페이지 {source['page']} "
                f"({source['type']}) "
                f"{' - ' + source['pdf_name'] if source.get('pdf_name') else ''}"
            )

        if result["images"]:
            print(f"\n이미지 {len(result['images'])}개:")
            for img_path in result["images"][:5]:
                print(f"  - {img_path}")

        if result["tables"]:
            print(f"\n테이블 {len(result['tables'])}개:")
            for table_path in result["tables"][:5]:
                print(f"  - {table_path}")

        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        print()


if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 서버 모드 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # FastAPI 서버 모드
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from pydantic import BaseModel
        from contextlib import asynccontextmanager
        import uvicorn
        import logging
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # QA 시스템 인스턴스 (전역으로 한 번만 로드)
        qa_system = None
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            global qa_system
            logger.info("Loading HybridQASystem...")
            try:
                qa_system = HybridQASystem()
                logger.info("System ready!")
            except Exception as e:
                logger.error(f"Failed to load HybridQASystem: {e}")
                raise
            yield
            # Shutdown
            logger.info("Shutting down...")
        
        app = FastAPI(title="Manual LLM API", lifespan=lifespan)
        
        # CORS 설정 (Next.js 개발 서버 포함)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3001",
                "http://localhost:3000",
                "http://127.0.0.1:3001",
                "http://127.0.0.1:3000",
                "*"  # 개발 환경용
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        class QuestionRequest(BaseModel):
            question: str
        
        @app.post("/ask")
        async def ask_question(request: QuestionRequest):
            """질문에 대한 답변 생성"""
            try:
                if not request.question or not request.question.strip():
                    raise HTTPException(
                        status_code=400, 
                        detail="질문을 입력해주세요."
                    )
                
                if qa_system is None:
                    raise HTTPException(
                        status_code=503,
                        detail="QA 시스템이 아직 초기화되지 않았습니다."
                    )
                
                logger.info(f"Received question: {request.question[:50]}...")
                result = qa_system.ask(request.question.strip())
                logger.info(f"Answer generated successfully. Sources: {result.get('num_sources', 0)}")
                
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing question: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"서버 오류가 발생했습니다: {str(e)}"
                )
        
        @app.get("/health")
        async def health():
            """헬스 체크 엔드포인트"""
            return {
                "status": "ok",
                "qa_system_loaded": qa_system is not None
            }
        
        @app.get("/images/{filename:path}")
        async def get_image(filename: str):
            """이미지 파일 서빙"""
            try:
                # 이미지 경로 찾기
                image_path = None
                if qa_system and hasattr(qa_system, 'output_dir'):
                    # images 디렉토리에서 찾기
                    image_path = qa_system.output_dir / "images" / filename
                    if not image_path.exists():
                        # tables 디렉토리에서도 찾기
                        image_path = qa_system.output_dir / "tables" / filename
                
                if not image_path or not image_path.exists():
                    raise HTTPException(status_code=404, detail="Image not found")
                
                return FileResponse(str(image_path))
            except Exception as e:
                logger.error(f"Error serving image {filename}: {e}")
                raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")
        
        # 포트 설정 (환경변수로 오버라이드 가능)
        import os
        port = int(os.getenv("PORT", 8003))
        logger.info(f"Starting Manual LLM API Server on http://0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # 기존 main() 함수 실행 (일반 모드)
        main()
