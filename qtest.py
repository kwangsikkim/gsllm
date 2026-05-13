"""
하이브리드 검색 기반 질문-답변 시스템
- 텍스트 기반 검색 (BM25) 50개 + 벡터 검색 (RAG) 50개
- 리랭커로 상위 10개 선택
- LLM으로 답변 생성
- 이미지/테이블 포함 시 경로 제공
"""

# ============================================================================
# 질문 작성 영역 - 여기에 질문을 작성하세요
# ============================================================================
QUESTION = """
측면의 지지조건 케이스 여섯개를 설명해줘
"""
# ============================================================================

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
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
        llm_model_path: str = LLM_MODEL_PATH
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
        # 리랭커 모델 로드
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
        self.reranker_model = AutoModel.from_pretrained(reranker_name)
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
            n_ctx=4096,   # 컨텍스트 길이
            n_threads=4,  # 스레드 수
            n_gpu_layers=n_gpu_layers,  # GPU 레이어 수
            verbose=False
        )
        
        # ChromaDB 연결
        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=str(self.output_dir / "chroma_db")
        )
        self.collection = self.client.get_collection("manual_embeddings")
        
        # BM25를 위한 문서 로드
        print("Loading documents for BM25...")
        self._load_documents_for_bm25()
        
        print("System ready!")
    
    def _load_documents_for_bm25(self):
        """BM25 검색을 위해 모든 문서 로드"""
        # ChromaDB에서 모든 문서 가져오기
        all_data = self.collection.get()
        
        self.documents = all_data.get("documents", [])
        self.metadatas = all_data.get("metadatas", [])
        self.ids = all_data.get("ids", [])
        
        # BM25를 위한 토큰화된 문서 (한국어 대응: 문자 bi-gram 토큰 사용)
        tokenized_docs = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def embed_text(self, text: str) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        with torch.no_grad():
            inputs = self.embed_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            # GPU로 입력 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.embed_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0]
    
    def text_search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """BM25 기반 텍스트 검색"""
        # 한국어 대응 토크나이저 사용
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # 상위 k개 선택
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            # 점수가 0이어도 상대적 순위는 의미 있으니 그대로 사용
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadatas[idx],
                "id": self.ids[idx],
                "score": float(scores[idx]),
                "search_type": "text"
            })
        
        return results
    
    def vector_search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """벡터 기반 검색 (RAG)"""
        query_embedding = self.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        formatted_results: List[Dict[str, Any]] = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                    "search_type": "vector"
                })
        
        return formatted_results
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
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
                batch_pairs = pairs[i:i+batch_size]
                batch_queries = [pair[0] for pair in batch_pairs]
                batch_docs = [pair[1] for pair in batch_pairs]
                
                # 토큰화
                inputs = self.reranker_tokenizer(
                    batch_queries,
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # GPU로 입력 이동
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.reranker_model(**inputs)
                # BGE reranker는 logits를 반환
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
    
    def hybrid_search(self, query: str, text_k: int = 50, vector_k: int = 50, final_k: int = 10) -> List[Dict[str, Any]]:
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
    
    def extract_images_and_tables(self, results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """검색 결과에서 이미지와 테이블 경로 추출 (사이드바 이미지 제외)"""
        images: List[str] = []
        tables: List[str] = []
        
        for result in results:
            metadata = result.get("metadata", {})
            element_type = metadata.get("type", "")
            path = metadata.get("path", "")
            is_sidebar = metadata.get("is_sidebar", False)
            
            if element_type == "image" and path and os.path.exists(path):
                # 사이드바 이미지는 제외
                if not is_sidebar:
                    images.append(path)
            elif element_type == "table" and path and os.path.exists(path):
                tables.append(path)
        
        return {"images": images, "tables": tables}
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """검색 결과를 LLM 입력용 컨텍스트로 포맷팅"""
        context_parts: List[str] = []
        
        for i, result in enumerate(results, 1):
            doc = result.get("document", "")
            metadata = result.get("metadata", {})
            page = metadata.get("page", "?")
            element_type = metadata.get("type", "text")
            
            context_parts.append(f"[문서 {i}] (페이지 {page}, 유형: {element_type})\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str, images: List[str], tables: List[str]) -> str:
        """LLM을 사용하여 답변 생성"""
        # 프롬프트 구성
        prompt = f"""당신은 기술 문서를 분석하여 질문에 답변하는 전문가입니다. 
제공된 문서를 정확하게 분석하고, 질문에 대한 명확하고 상세한 답변을 한글로 작성해주세요.

참고 문서:
{context}

"""
        
        # 이미지나 테이블이 있는 경우 언급
        if images or tables:
            prompt += "중요 참고사항:\n"
            if images:
                prompt += f"- 관련 이미지 {len(images)}개가 있습니다. 이미지의 내용을 참고하여 답변하세요.\n"
            if tables:
                prompt += f"- 관련 테이블 {len(tables)}개가 있습니다. 테이블의 수치와 데이터를 정확하게 참고하여 답변하세요.\n"
            prompt += "\n"
        
        prompt += f"질문: {query}\n\n"
        prompt += """답변 작성 지침:
1. 반드시 한글로 작성하세요.
2. 참고 문서의 내용을 정확하게 인용하세요.
3. 테이블이나 이미지가 있다면 그 내용을 구체적으로 언급하세요.
4. 수치나 데이터가 있다면 정확하게 제시하세요.
5. 불확실한 내용은 추측하지 말고 문서에 명시된 내용만 답변하세요.

답변:"""
        
        # LLM으로 답변 생성
        response = self.llm(
            prompt,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.15,
            stop=["질문:", "[문서", "참고 문서:"]
        )
        
        answer = response["choices"][0]["text"].strip()
        
        # 답변이 잘렸는지 판단
        is_truncated = False
        finish_reason = response["choices"][0].get("finish_reason", "")
        
        if finish_reason == "length":
            is_truncated = True
        
        if answer and not answer[-1] in ['.', '!', '?', '다', '요', '니다', '습니다', '니다.', '습니다.', '요.', '다.']:
            if len(answer) > 100:
                is_truncated = True
        
        if len(answer) < 20:
            answer = "참고 문서를 기반으로 답변을 생성했지만 내용이 충분하지 않습니다. 더 많은 컨텍스트가 필요할 수 있습니다."
        elif is_truncated:
            answer += "\n\n[참고: 답변이 토큰 제한으로 인해 잘렸을 수 있습니다. 더 자세한 정보가 필요하면 질문을 더 구체적으로 해주세요.]"
        
        return answer
    
    def ask(self, query: str) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        results = self.hybrid_search(query, text_k=50, vector_k=50, final_k=10)
        
        if not results:
            return {
                "query": query,
                "answer": "관련 문서를 찾을 수 없습니다.",
                "sources": [],
                "images": [],
                "tables": []
            }
        
        media = self.extract_images_and_tables(results)
        context = self.format_context(results)
        
        print("\n답변 생성 중...")
        answer = self.generate_answer(query, context, media["images"], media["tables"])
        
        sources = []
        for result in results:
            metadata = result.get("metadata", {})
            sources.append({
                "page": metadata.get("page", "?"),
                "type": metadata.get("type", "text"),
                "section": metadata.get("section", ""),
                "rerank_score": result.get("rerank_score", 0)
            })
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "images": media["images"],
            "tables": media["tables"],
            "num_sources": len(results)
        }


def main():
    """질문-답변 테스트"""
    qa_system = HybridQASystem()
    
    query = QUESTION.strip()
    
    default_texts = [
        "여기에 질문을 작성하세요.",
        "여기에 질문을 작성하세요.\n예: 사용 방법은 무엇인가요?",
        ""
    ]
    
    if not query or query in default_texts:
        print("오류: QUESTION 변수에 질문을 작성해주세요.")
        print("파일 상단의 QUESTION = \"\"\" ... \"\"\" 부분에 질문을 입력하세요.")
        return
    
    print("\n" + "="*60)
    print("질문-답변 시스템")
    print("="*60)
    print(f"\n질문: {query}\n")
    
    try:
        result = qa_system.ask(query)
        
        print("\n" + "="*60)
        print("답변:")
        answer_text = result["answer"]
        
        if "[참고: 답변이 토큰 제한으로 인해 잘렸을 수 있습니다" in answer_text:
            print(answer_text)
            print("\n⚠️  경고: 답변이 토큰 제한으로 인해 잘렸을 수 있습니다.")
        else:
            print(answer_text)
        
        print("\n" + "-"*60)
        print(f"참고 문서: {result['num_sources']}개")
        for i, source in enumerate(result["sources"][:5], 1):
            print(f"  {i}. 페이지 {source['page']} ({source['type']})")
        
        if result["images"]:
            print(f"\n이미지 {len(result['images'])}개:")
            for img_path in result["images"][:3]:
                print(f"  - {img_path}")
        
        if result["tables"]:
            print(f"\n테이블 {len(result['tables'])}개:")
            for table_path in result["tables"][:3]:
                print(f"  - {table_path}")
        
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
