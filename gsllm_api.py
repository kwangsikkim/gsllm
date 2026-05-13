#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gsllm HTTP API — embed_docs.run_embed_job, embed_query_all.run_single_query 연동.

주요 경로:
  POST /gsllm/query, POST /gsllm/chat (form query=)
  — 챗봇 프록시(예: Next chatmcp)는 /gsllm/query JSON에 with_answer·model_id를 명시하는 것을 권장(서버 기본값만 믿지 않음).
  GET  /gsllm/documents?email=&role=&store=<이름> 또는 chroma_dir=<절대경로>
  DELETE /gsllm/documents/embed — 컬렉션 내 특정 문서(청크·data 파일·work/page_images) 제거.
    쿼리: doc_key (권장, 빠름) 또는 storage_basename / filename 과 택일
  POST /gsllm/documents/delete — 위와 동일(본문 JSON, 긴 파일명에 유리)
  GET/POST/DELETE /collections (emb.py 호환, rag-collections 프록시용)

실행 (저장소 루트에서):
  uvicorn gsllm_api:app --host 0.0.0.0 --port 8010

환경 변수:
  GSLLM_PATH_ALLOW_PREFIX — 허용 경로(복수는 os.pathsep). 기본: gsllm 루트 + /home/siwasoft/siwasoft/mcp/pdf
  GSLLM_CHROMA_BASE — 컬렉션 목록/생성 시 부모 경로 (기본: embed_query_all.DEFAULT_CHROMA_BASE, 보통 .../embed_test/chroma)
  GSLLM_DATA_BASE — 업로드 원본 삭제 시 부모 (기본: …/embed_test/data). Next GSLLM_LOCAL_UPLOAD_DIR 와 맞출 것.
  GSLLM_WORK_BASE — page_images 정리 시 부모 (기본: …/embed_test/work)
  GSLLM_EMBED_WORK_DIR / GSLLM_EMBED_CHROMA_DIR — 임베딩 시 work·chroma 고정(미설정 시 work/<컬렉션명>, chroma_base/<컬렉션명>)
  GSLLM_EMBED_MODEL / GSLLM_RERANKER_MODEL — 질의 API startup 시 로드할 모델 (기본: embed_query_all 상수)
  GSLLM_QUERY_DEVICE — cuda | cpu | 비우면 자동
  GSLLM_SOFFICE — LibreOffice 실행 파일 경로 (미설정 시 PATH에서 soffice/libreoffice 검색)
  GSLLM_CONVERT_TIMEOUT_SEC — Office→PDF 변환 타임아웃 초 (기본 300)
  GSLLM_EMBED_JOB_TTL_SEC — 종료된 임베딩 job 상태 보관 시간(초, 기본 86400). job 레코드는 프로세스 메모리에만 있음(아래 임베딩 API 참고).
  GSLLM_CHROMA_SUBPROCESS_TIMEOUT_SEC — /collections Chroma 자식 프로세스 타임아웃(초, 기본 60)
  GSLLM_PUBLIC_BASE_URL — 질의 응답의 source_pdf_url·마크다운 링크에 쓸 공개 베이스 URL (미설정 시 요청 Host의 base_url)
  GSLLM_APPEND_SOURCE_LINKS — 답변(answer) 끝에 원본 PDF 링크 마크다운 블록 추가 (기본 1, 끄려면 0)
  GSLLM_SOURCE_LINKS_MODE — 링크 블록에 넣을 문서 범위: cited(기본, 본문 [출처:…]에 맞는 항목만) | retrieval(검색 상위 전체) | both(「답변 근거」+「검색 참고 문서」 두 절)
  GSLLM_PLAIN_ANSWER — LLM 답변 본문만 마크다운 제거(평문화). 원본 PDF 블록(--- 이하)은 유지 (기본 1, 끄려면 0)
  GSLLM_LINKIFY_CITATIONS — 본문의 [출처: …] 및 깨진 '… p.N]' 줄을 source-pdf URL 마크다운 링크로 치환 (기본 1, 끄려면 0)
  GSLLM_ANSWER_READABILITY — 답변 본문에서 p.N"] 류 오타 정리·문장과 출처(파일명) 줄 분리·[출처: 앞 줄바꿈 (기본 1, 끄려면 0)

  GSLLM_EMBED_REPLACE_SAME_ORIGINAL — 임베딩 전에 동일 metadata.original_filename 청크 Chroma 에서 삭제(기본 1, 끄려면 0)

  GET /gsllm/documents — 임베딩 문서 목록(파일 단위 묶음). 쿼리 `email`·`role` 필수, store/chroma_dir 기준 RBAC.
  응답 각 문서 항목에 doc_key 포함 (임베딩 후).
  문서 삭제 시 storage_basename(디스크 PDF 베이스명) 또는 doc_key 권장.

doc_key 간단 검증 시나리오:
  1) POST /gsllm/embed(또는 embed-jobs)로 PDF 1건 임베딩
  2) GET /gsllm/documents 로 목록 조회 후 항목의 doc_key 확인
  3) DELETE .../documents/embed 에 doc_key 로 삭제하면 해당 논리 문서 청크만 제거
  4) 동일 원본 이름으로 재업로드·임베딩 시 (기본) 기존 original_filename 청크가 미리 삭제되어 중복 유지 안 됨

Chroma 동시 접근: 같은 uvicorn 프로세스에서 persist 경로(예: …/chroma/collection_B)당 뮤텍스로
  임베딩 업서트·purge·GET /gsllm/documents·문서 삭제 API 가 직렬화되어 Component not running 를 줄인다.

임베딩 API (웹·프록시는 비동기 job 경로 권장):
  - 권장: POST /gsllm/embed-jobs (또는 /embed-jobs) → 즉시 `{ ok, job_id, status: "queued" }` 반환. 실제 임베딩은 백그라운드 스레드에서 수행.
  - 폴링: GET /gsllm/embed-jobs/{job_id} (또는 /embed-jobs/{job_id}). 응답에는 항상 문자열 `status` 가 포함됨: queued | running | succeeded | failed | cancelled | unknown(비정상).
    진행 중(queued/running 등)에는 JSON 필드 `ok` 가 false 일 수 있음 — UI·클라이언트는 `status` 로 성공/실패를 판단할 것(`ok` 는 succeeded 여부만 의미).
    성공 시에만 `result`, 실패 시에만 `error` 가 포함될 수 있음. `progress` 는 있을 때 단계 메시지.
  - 동기: POST /gsllm/embed — 응답이 임베딩 완료까지 열려 있어 HTTP/프록시 타임아웃과 맞지 않기 쉬움. curl·내부 스크립트용. siwasoftwebtest 등 브라우저 경로의 기본값으로 쓰지 말 것.
  - 운영: uvicorn 워커가 여러 프로세스면 job_id는 생성한 그 워커에서만 조회 가능(메모리 저장). 단일 워커 또는 `--workers 1` 권장. 재시작 시 job 목록은 초기화됨.

emb.py 호환: GET/POST/DELETE `/collections` — siwasoftwebtest `rag-collections.js` 가 GSLLM_API_BASE(8010)만 쓸 때 연동.
  Chroma 생성·삭제·목록은 `chroma_subprocess.py` 를 subprocess로 호출해 프로세스 종료 시 클라이언트 상태가 남지 않게 함.
  GET `/collections` (다중 스토어 모드): `chroma.sqlite3` 없이 빈 스토어 폴더만 있어도, `chroma/`·`data/` 하위 디렉터리 이름을 합쳐 목록에 포함
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import traceback
import unicodedata
import shutil
import subprocess
import sys
import time
import uuid
import gc
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

import chromadb
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import embed_docs
import embed_query_all


_REPO_ROOT = Path(__file__).resolve().parent
# 웹 업로드 서버가 PDF를 두는 흔한 경로( siwasoftwebtest rag-embedding 의 PDF_DIR 기본과 맞춤 )
_DEFAULT_UPLOAD_PDF = Path("/home/siwasoft/siwasoft/mcp/pdf")

# LibreOffice --convert-to pdf 지원 확장자 (소문자 기준)
_OFFICE_TO_PDF_EXT = frozenset({
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
})


def _find_soffice() -> Optional[str]:
    env = (os.environ.get("GSLLM_SOFFICE") or "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _convert_office_to_pdf(src: Path, *, timeout_sec: int) -> Path:
    """
    원본과 같은 디렉터리에 `<stem>.pdf` 생성 (LibreOffice 무헤드).
    """
    soffice = _find_soffice()
    if not soffice:
        raise RuntimeError(
            "LibreOffice를 찾을 수 없습니다. apt 등으로 설치하거나 GSLLM_SOFFICE에 soffice 경로를 지정하세요."
        )
    src = src.resolve()
    if not src.is_file():
        raise FileNotFoundError(str(src))
    out_dir = src.parent
    expected = out_dir / f"{src.stem}.pdf"
    cmd = [
        soffice,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(src),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"LibreOffice 변환 실패 (exit {proc.returncode}): {tail[:2000]}"
        )
    if expected.is_file():
        return expected
    # 비ASCII 파일명 등에서 이름이 달라질 때 대비
    matches = sorted(
        out_dir.glob(f"{src.stem}*.pdf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    raise RuntimeError(f"변환 후 PDF를 찾지 못했습니다: 기대 경로 {expected}")


def _allow_prefixes() -> List[Path]:
    default_raw = f"{_REPO_ROOT}{os.pathsep}{_DEFAULT_UPLOAD_PDF}"
    raw = os.environ.get("GSLLM_PATH_ALLOW_PREFIX", default_raw).strip()
    parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
    if not parts:
        parts = [str(_REPO_ROOT)]
    return [Path(p).expanduser().resolve() for p in parts]


def _is_allowed_path(path: Path, bases: List[Path]) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return False
    for base in bases:
        try:
            resolved.relative_to(base)
            return True
        except ValueError:
            continue
    return False


def _default_chroma_parent() -> Path:
    return Path(
        os.environ.get("GSLLM_CHROMA_BASE", str(embed_query_all.DEFAULT_CHROMA_BASE)),
    ).expanduser().resolve()


def _default_data_parent() -> Path:
    return Path(
        os.environ.get("GSLLM_DATA_BASE", str(_REPO_ROOT / "embed_test" / "data")),
    ).expanduser().resolve()


def _default_work_parent() -> Path:
    return Path(
        os.environ.get("GSLLM_WORK_BASE", str(_REPO_ROOT / "embed_test" / "work")),
    ).expanduser().resolve()


def _chroma_subprocess_timeout_sec() -> int:
    raw = (os.environ.get("GSLLM_CHROMA_SUBPROCESS_TIMEOUT_SEC", "60") or "").strip()
    try:
        return max(5, int(raw))
    except ValueError:
        return 60


def _run_chroma_subprocess(
    action: str,
    *,
    path: Path,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """Chroma 전용 자식 프로세스 — 작업 종료 시 프로세스 메모리·캐시가 함께 사라진다."""
    script = (_REPO_ROOT / "chroma_subprocess.py").resolve()
    if not script.is_file():
        raise RuntimeError(f"chroma subprocess script not found: {script}")

    tsec = timeout_sec if timeout_sec is not None else _chroma_subprocess_timeout_sec()
    cmd: List[str] = [
        sys.executable,
        str(script),
        action,
        "--path",
        str(path),
    ]
    if name:
        cmd += ["--name", name]
    if metadata is not None:
        cmd += ["--metadata-json", json.dumps(metadata, ensure_ascii=False)]

    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=tsec,
    )

    stdout = (proc.stdout or "").strip()
    last_line = ""
    if stdout:
        last_line = stdout.splitlines()[-1].strip()

    def _fail_detail() -> str:
        if last_line:
            try:
                obj = json.loads(last_line)
                if isinstance(obj, dict) and obj.get("error"):
                    return str(obj["error"])
            except Exception:
                pass
        tail = (proc.stderr or proc.stdout or "").strip()
        return tail[:4000] if tail else "(no output)"

    if proc.returncode != 0:
        raise RuntimeError(
            f"chroma subprocess failed action={action} path={path}: {_fail_detail()}"
        )

    if not last_line:
        raise RuntimeError(
            f"chroma subprocess returned empty stdout action={action} path={path}"
        )

    try:
        out = json.loads(last_line)
    except Exception as e:
        raise RuntimeError(
            f"failed to parse chroma subprocess JSON action={action} path={path}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        ) from e

    if not isinstance(out, dict) or not out.get("ok"):
        raise RuntimeError(
            f"chroma subprocess returned not ok: {out!r}"
        )

    return out


def _resolve_chroma_arg(chroma: Optional[str]) -> Path:
    if chroma and str(chroma).strip():
        return Path(str(chroma).strip()).expanduser().resolve()
    return _default_chroma_parent()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _embed_jobs_ttl_sec() -> int:
    raw = (os.environ.get("GSLLM_EMBED_JOB_TTL_SEC", "86400") or "").strip()
    try:
        sec = int(raw)
    except ValueError:
        sec = 86400
    return max(300, sec)


def _prune_embed_jobs(now: Optional[float] = None) -> None:
    ts_now = now if now is not None else time.time()
    ttl = _embed_jobs_ttl_sec()
    stale: List[str] = []
    with _EMBED_JOBS_LOCK:
        for jid, rec in _EMBED_JOBS.items():
            done = rec.get("status") in {"succeeded", "failed", "cancelled"}
            updated_ts = float(rec.get("updated_ts") or rec.get("created_ts") or 0.0)
            if done and (ts_now - updated_ts) > ttl:
                stale.append(jid)
        for jid in stale:
            _EMBED_JOBS.pop(jid, None)


def _job_public_view(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET embed-jobs 공개 응답. `status` 는 항상 문자열(queued|running|succeeded|failed|unknown).
    `ok` 는 succeeded 일 때만 True — 진행 중에는 False 이므로 클라이언트는 status 로 판단한다.
    """
    raw_status = rec.get("status")
    status_s = str(raw_status or "").strip().lower()
    if status_s not in ("queued", "running", "succeeded", "failed", "cancelled"):
        status_s = "unknown"
    out: Dict[str, Any] = {
        "ok": status_s == "succeeded",
        "job_id": rec.get("job_id"),
        "status": status_s,
        "created_at": rec.get("created_at"),
        "updated_at": rec.get("updated_at"),
    }
    progress = rec.get("progress")
    if isinstance(progress, dict) and progress:
        out["progress"] = progress
    if rec.get("status") == "succeeded" and isinstance(rec.get("result"), dict):
        out["result"] = rec["result"]
    if rec.get("status") == "failed" and isinstance(rec.get("error"), dict):
        out["error"] = rec["error"]
    return out


def _is_chroma_readonly_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "readonly" in msg or "read-only" in msg


def _collection_lock(name: str) -> threading.Lock:
    k = str(name or "").strip().lower()
    with _COLLECTION_LOCKS_GUARD:
        lock = _COLLECTION_LOCKS.get(k)
        if lock is None:
            lock = threading.Lock()
            _COLLECTION_LOCKS[k] = lock
        return lock


def _collection_delete_cooldown_sec() -> float:
    raw = (os.environ.get("GSLLM_COLLECTION_DELETE_COOLDOWN_SEC", "1.5") or "").strip()
    try:
        v = float(raw)
    except ValueError:
        v = 1.5
    return max(0.0, v)


def _wait_if_recently_deleted(name: str) -> None:
    k = str(name or "").strip().lower()
    if not k:
        return
    cooldown = _collection_delete_cooldown_sec()
    with _COLLECTION_LOCKS_GUARD:
        last = _COLLECTION_LAST_DELETE_TS.get(k)
    if last is None:
        return
    elapsed = time.time() - last
    remaining = cooldown - elapsed
    if remaining > 0:
        # short backoff to let chromadb/sqlite handles settle after delete
        time.sleep(remaining)


def _mark_collection_deleted(name: str) -> None:
    k = str(name or "").strip().lower()
    if not k:
        return
    with _COLLECTION_LOCKS_GUARD:
        _COLLECTION_LAST_DELETE_TS[k] = time.time()


def _clear_chroma_system_cache() -> None:
    """
    chromadb 0.4.x는 같은 프로세스 안에서 Settings 식별자별 System을 전역 캐시한다.
    컬렉션 디렉터리를 삭제한 뒤 같은 경로를 즉시 재사용하면, 이 캐시가 이전 sqlite/sysdb
    상태를 붙잡아 `no such table: tenants`가 반복될 수 있어 best-effort로 비운다.
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        cache = getattr(SharedSystemClient, "_identifier_to_system", None)
        if isinstance(cache, dict):
            for sys_obj in list(cache.values()):
                try:
                    stop = getattr(sys_obj, "stop", None)
                    if callable(stop):
                        stop()
                except Exception:
                    pass
            cache.clear()
    except Exception:
        pass


def _persistent_chroma_client(path: Path | str) -> Any:
    """
    embed_docs·embed_query_all과 동일하게 PersistentClient만 사용한다.
    같은 persist 경로에 chromadb.Client(Settings(...))를 섞으면 전역 SharedSystemClient가
    \"different settings\" ValueError를 낸다(임베딩 스레드가 PersistentClient를 연 뒤 문서 목록 API가 터짐).
    """
    p = Path(path).expanduser().resolve()
    return chromadb.PersistentClient(path=str(p))


# 같은 persist 디렉터리에 대한 임베딩(업서트)·purge·목록·삭제를 프로세스 내에서 직렬화한다.
# 동시에 PersistentClient 가 열리면 Chroma 에서 RuntimeError("Component not running") 가 날 수 있음.
_CHROMA_PERSIST_LOCK_GUARD = threading.Lock()
_CHROMA_PERSIST_LOCKS: Dict[str, threading.Lock] = {}


def _chroma_persist_lock(path: Path | str) -> threading.Lock:
    key = str(Path(path).expanduser().resolve())
    with _CHROMA_PERSIST_LOCK_GUARD:
        if key not in _CHROMA_PERSIST_LOCKS:
            _CHROMA_PERSIST_LOCKS[key] = threading.Lock()
        return _CHROMA_PERSIST_LOCKS[key]


def _close_chroma_client(client: Any, *, clear_cache: bool = False) -> None:
    """Best-effort close for chromadb PersistentClient to release sqlite/file handles."""
    if client is None:
        return
    try:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception:
        pass
    if clear_cache:
        _clear_chroma_system_cache()

def _safe_rmtree(target: Path, *, label: str) -> bool:
    """
    Remove a directory and verify it is gone.
    Returns True when a directory existed and was deleted.
    Raises HTTPException when deletion fails.
    """
    if not target.exists():
        return False
    if not target.is_dir():
        raise HTTPException(status_code=500, detail=f"{label} is not a directory: {target}")
    try:
        shutil.rmtree(target)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"failed to delete {label}: {target} ({e})",
        ) from e
    if target.exists():
        raise HTTPException(status_code=500, detail=f"failed to delete {label}: {target} (still exists)")
    return True


_query_embedder = None
_query_reranker = None
_query_device: Optional[str] = None
_query_embed_id: Optional[str] = None
_query_reranker_id: Optional[str] = None

# embed-jobs 전용. 프로세스 메모리만 사용 — uvicorn --workers N>1 이면 생성한 워커가 아닌 곳에서 GET 시 404 가능.
_EMBED_JOBS: Dict[str, Dict[str, Any]] = {}
_EMBED_JOBS_LOCK = threading.Lock()
_COLLECTION_LOCKS: Dict[str, threading.Lock] = {}
_COLLECTION_LOCKS_GUARD = threading.Lock()
_COLLECTION_LAST_DELETE_TS: Dict[str, float] = {}


_ADMIN_ROLE = "admin"
_ROLE_ALLOWED_COLLECTIONS: Dict[str, set[str]] = {
    "team_a": {"collection_A"},
    "team_b": {"collection_B"},
}
_ROLE_EMAIL_MAP: Dict[str, str] = {
    "admin": "haetae@test.com",
    "team_a": "itai@ht.co.kr",
    "team_b": "siwa@ht.co.kr",
}


def _norm_role(role: Optional[str]) -> str:
    return str(role or "").strip().lower()


def _norm_email(email: Optional[str]) -> str:
    return str(email or "").strip().lower()


def _authz_error(role: str, target: str) -> HTTPException:
    return HTTPException(
        status_code=403,
        detail=f"forbidden: role {role or '(unknown)'} cannot access collection {target}",
    )


def _assert_role_email(email: Optional[str], role: Optional[str]) -> tuple[str, str]:
    role_n = _norm_role(role)
    email_n = _norm_email(email)
    if not role_n:
        raise HTTPException(status_code=400, detail="role is required")
    if not email_n:
        raise HTTPException(status_code=400, detail="email is required")
    expected = _ROLE_EMAIL_MAP.get(role_n)
    if expected and email_n != expected:
        raise HTTPException(
            status_code=403,
            detail=f"forbidden: role {role_n} does not match email {email_n}",
        )
    if role_n != _ADMIN_ROLE and role_n not in _ROLE_ALLOWED_COLLECTIONS:
        raise HTTPException(status_code=403, detail=f"forbidden: unknown role {role_n}")
    return email_n, role_n


def _allowed_collections(role_n: str) -> Optional[set[str]]:
    if role_n == _ADMIN_ROLE:
        return None
    return set(_ROLE_ALLOWED_COLLECTIONS.get(role_n, set()))


def _enforce_collection_access(
    *,
    email: Optional[str],
    role: Optional[str],
    requested_collection: Optional[str],
    endpoint: str,
    require_collection: bool = False,
) -> tuple[str, str, Optional[str]]:
    email_n, role_n = _assert_role_email(email=email, role=role)
    req = str(requested_collection or "").strip() or None
    allowed = _allowed_collections(role_n)
    if allowed is None:
        if require_collection and not req:
            raise HTTPException(status_code=400, detail="collection is required")
        eff = req
    else:
        if req:
            if req not in allowed:
                raise _authz_error(role_n, req)
            eff = req
        else:
            # 팀 계정은 collection 미지정 시 자기 팀 컬렉션으로 강제
            eff = sorted(allowed)[0] if allowed else None
        if require_collection and not eff:
            raise HTTPException(status_code=400, detail="collection is required")
    print(
        f"[AUTHZ] endpoint={endpoint} email={email_n} role={role_n} "
        f"requested_collection={req or '-'} effective_collection={eff or '-'}"
    )
    return email_n, role_n, eff


def _source_pdf_public_base(request: Request) -> str:
    env = (os.environ.get("GSLLM_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if env:
        return env
    return str(request.base_url).rstrip("/")


def _build_source_pdf_url(
    public_base: str,
    email: str,
    role: str,
    store: str,
    file: str,
    page: Optional[int],
    as_display: Optional[str] = None,
) -> str:
    q = (
        f"email={quote(email, safe='')}"
        f"&role={quote(role, safe='')}"
        f"&store={quote(store, safe='')}"
        f"&file={quote(file, safe='')}"
    )
    disp = (as_display or "").strip()
    if disp:
        q += f"&as={quote(disp, safe='')}"
    frag = f"#page={int(page)}" if page is not None else ""
    return f"{public_base}/gsllm/source-pdf?{q}{frag}"


def _enrich_source_pdf_links(
    payload: Dict[str, Any],
    *,
    public_base: str,
    email: str,
    role: str,
) -> None:
    """sources[].source_pdf_url 및 source_page_links(중복 제거) 설정."""
    links: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, Optional[int]]] = set()
    for s in payload.get("sources") or []:
        if not isinstance(s, dict):
            continue
        sp = s.get("source_pdf")
        if not isinstance(sp, dict):
            continue
        store = str(sp.get("store") or "").strip()
        file = str(sp.get("file") or "").strip()
        raw_page = sp.get("page")
        p_int: Optional[int] = None
        if raw_page is not None:
            try:
                p_int = int(raw_page)
            except (TypeError, ValueError):
                p_int = None
        if not store or not file:
            continue
        meta = s.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        as_disp = str(meta.get("original_filename") or "").strip() or None
        url = _build_source_pdf_url(
            public_base, email, role, store, file, p_int, as_display=as_disp
        )
        s["source_pdf_url"] = url
        key = (store, file, p_int)
        if key in seen:
            continue
        seen.add(key)
        label = str(s.get("source_file") or file).strip()
        ptxt = f" p.{p_int}" if p_int is not None else ""
        links.append({"label": (label + ptxt).strip(), "url": url})
    payload["source_page_links"] = links
    payload["source_page_links_retrieval"] = [dict(x) for x in links]


def _source_links_mode() -> str:
    raw = (os.environ.get("GSLLM_SOURCE_LINKS_MODE", "cited") or "cited").strip().lower()
    if raw in ("retrieval", "cited", "both"):
        return raw
    return "cited"


def _extract_citation_labels_from_answer(text: str) -> List[str]:
    """본문에서 인용 라벨 수집(linkify 전·후 형식 모두). 순서 유지, 정규화 키 기준 중복 제거."""
    t = str(text or "")
    seen_norm: set[str] = set()
    out: List[str] = []

    def add_raw(raw: str) -> None:
        lab = (raw or "").strip()
        if not lab:
            return
        nk = _normalize_citation_key(lab)
        if nk in seen_norm:
            return
        seen_norm.add(nk)
        out.append(lab)

    for m in re.finditer(r"(?m)^\s*출처:\s*\[([^\]]+)\]\([^)]+\)\s*$", t):
        add_raw(m.group(1))
    for m in re.finditer(r"(?m)^\s*\[출처:\s*([^\]]+?)\]\s*$", t):
        add_raw(m.group(1))
    for m in re.finditer(r"\[출처:\s*([^\]]+?)\]", t):
        add_raw(m.group(1))
    return out


def _link_label_matches_citations(link_label: str, cited_norms: set[str]) -> bool:
    if not cited_norms:
        return False
    k = _normalize_citation_key(link_label)
    if not k:
        return False
    if k in cited_norms:
        return True
    for c in cited_norms:
        if len(c) < 8:
            continue
        if c in k or (len(k) >= 8 and k in c):
            return True
    return False


def _finalize_source_page_links_for_append(payload: Dict[str, Any]) -> None:
    """
    링크ify 이후 answer에서 인용을 파싱해 source_page_links_cited를 채우고,
    GSLLM_SOURCE_LINKS_MODE에 따라 source_page_links를 맞춘다.
    """
    ans = str(payload.get("answer") or "")
    retrieval: List[Dict[str, Any]] = []
    raw_r = payload.get("source_page_links_retrieval")
    if isinstance(raw_r, list) and raw_r:
        retrieval = [dict(x) for x in raw_r if isinstance(x, dict)]
    if not retrieval:
        raw_s = payload.get("source_page_links")
        if isinstance(raw_s, list):
            retrieval = [dict(x) for x in raw_s if isinstance(x, dict)]
            payload["source_page_links_retrieval"] = list(retrieval)

    labels = _extract_citation_labels_from_answer(ans)
    cited_norms = {_normalize_citation_key(x) for x in labels if str(x).strip()}

    seen_url: set[str] = set()
    cited_links: List[Dict[str, str]] = []
    for L in retrieval:
        url = str(L.get("url") or "").strip()
        if not _link_label_matches_citations(str(L.get("label") or ""), cited_norms):
            continue
        if url:
            if url in seen_url:
                continue
            seen_url.add(url)
        cited_links.append(dict(L))

    payload["source_page_links_cited"] = cited_links
    mode = _source_links_mode()
    if mode == "retrieval":
        payload["source_page_links"] = list(retrieval)
    elif mode == "cited":
        if not cited_links and not bool(payload.get("with_answer")):
            payload["source_page_links"] = list(retrieval)
        else:
            payload["source_page_links"] = list(cited_links)
    else:
        payload["source_page_links"] = list(cited_links)


def _markdown_source_link_label(label: str) -> str:
    """
    원본 PDF·본문 출처 링크 라벨 정규화. 백슬래시 이스케이프(\\[, \\])는
    경량 마크다운 렌더러에서 링크 전체가 깨지는 경우가 많아 쓰지 않는다.
    """
    s = str(label or "").strip()
    if not s:
        return "PDF"
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("|", "·")
    s = re.sub(r"\s+", " ", s)
    return s or "PDF"


def _normalize_citation_key(s: str) -> str:
    t = unicodedata.normalize("NFKC", str(s or ""))
    t = t.replace("\u00a0", " ").replace("\u202f", " ")
    t = re.sub(r"\s+", " ", t.strip())
    return t


def _linkify_citations_enabled() -> bool:
    return (os.environ.get("GSLLM_LINKIFY_CITATIONS", "1") or "").strip() != "0"


def _citation_readability_enabled() -> bool:
    return (os.environ.get("GSLLM_ANSWER_READABILITY", "1") or "").strip() != "0"


_CIT_STRAY_QUOTE_RE = re.compile(
    r"(\.(?:pptx|pdf))\s*(p\.\d+)\s*\"\s*\]?",
    re.IGNORECASE,
)


def _fix_citation_stray_quotes(s: str) -> str:
    return _CIT_STRAY_QUOTE_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}]", s)


def _split_inline_doc_citation_line(line: str) -> str:
    raw = line.rstrip("\r\n")
    if not raw.strip():
        return line
    stripped = raw.strip()
    if stripped.count("|") >= 2 and "|" in stripped:
        return line
    if not re.search(r"\.(?:pptx|pdf)\s+p\.\d+", raw, re.I):
        return line
    last_pos: Optional[int] = None
    for m in re.finditer(r"[\.。!？\?…](?:\s+)", raw):
        pos = m.end()
        tail = raw[pos:].strip()
        if not tail:
            continue
        if re.match(r"^[^\n]+\.(?:pptx|pdf)\s+p\.\d+", tail, re.I):
            last_pos = pos
    if last_pos is None:
        return line
    head = raw[:last_pos].rstrip()
    if not head:
        return line
    tail = raw[last_pos:].strip()
    ws = re.match(r"^(\s*)", line)
    indent = ws.group(1) if ws else ""
    return head + "\n\n" + indent + tail


def _apply_citation_readability_text(text: str) -> str:
    if not _citation_readability_enabled():
        return text
    text = _fix_citation_stray_quotes(text)
    lines = text.split("\n")
    text = "\n".join(_split_inline_doc_citation_line(L) for L in lines)
    text = re.sub(r"([\.。!？\?…])\s+(\[출처:)", r"\1\n\n\2", text)
    # Improve readability: after a citation line, separate the next top-level
    # numbered section with two blank lines (e.g., "출처..." -> "\n\n\n2. ...").
    text = re.sub(
        r"(?m)^(\s*(?:\[출처:[^\n]*\]|출처:\s*\[[^\n]*\]\([^\n]+\))\s*)\n(?=\s*\d+\.\s+)",
        r"\1\n\n\n",
        text,
    )
    return text


def _apply_citation_readability_payload(payload: Dict[str, Any]) -> None:
    if not _citation_readability_enabled():
        return
    ans = str(payload.get("answer") or "")
    if not ans.strip():
        return
    new = _apply_citation_readability_text(ans)
    payload["answer"] = new
    payload["message"] = new
    payload["response"] = new


def _sources_label_url_pairs(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in payload.get("sources") or []:
        if not isinstance(s, dict):
            continue
        url = str(s.get("source_pdf_url") or "").strip()
        if not url:
            continue
        sp = s.get("source_pdf")
        if not isinstance(sp, dict):
            continue
        file = str(sp.get("file") or "").strip()
        raw_page = sp.get("page")
        p_int: Optional[int] = None
        if raw_page is not None:
            try:
                p_int = int(raw_page)
            except (TypeError, ValueError):
                p_int = None
        label = str(s.get("source_file") or file).strip()
        ptxt = f" p.{p_int}" if p_int is not None else ""
        full = (label + ptxt).strip()
        if full:
            out.append((full, url))
    out.sort(key=lambda x: len(x[0]), reverse=True)
    return out


def _linkify_citations_in_answer(payload: Dict[str, Any]) -> None:
    if not _linkify_citations_enabled():
        return
    text = str(payload.get("answer") or "")
    if not text.strip():
        return
    pairs = _sources_label_url_pairs(payload)
    if not pairs:
        return
    norm_to_url: Dict[str, str] = {}
    for label, url in pairs:
        k = _normalize_citation_key(label)
        if k and k not in norm_to_url:
            norm_to_url[k] = url
    md_disp = _markdown_source_link_label
    for label, url in pairs:
        pat = r"(?m)^(\s*)\[출처:\s*" + re.escape(label) + r"\]"
        text = re.sub(
            pat,
            lambda m, u=url, lb=label: f"{m.group(1)}출처: [{md_disp(lb)}]({u})",
            text,
        )
    lines = text.split("\n")
    orphan_re = re.compile(r"^(\s*)([^\n]+?p\.\d+)\]\s*$")
    out_lines: List[str] = []
    for line in lines:
        if "[출처:" in line:
            out_lines.append(line)
            continue
        m = orphan_re.match(line.rstrip("\r"))
        if m:
            indent, body = m.group(1), m.group(2).strip()
            key = _normalize_citation_key(body)
            url = norm_to_url.get(key)
            if url is None and body.startswith("- "):
                url = norm_to_url.get(_normalize_citation_key(body[2:].strip()))
            if url:
                body_clean = body[2:].strip() if body.startswith("- ") else body
                md = _markdown_source_link_label(body_clean)
                line = f"{indent}출처: [{md}]({url})"
        out_lines.append(line)
    text = "\n".join(out_lines)
    text = re.sub(r"([\.。!？\?…])\s+(출처:\s*\[)", r"\1\n\n\2", text)
    text = re.sub(
        r"(?m)^(\s*(?:\[출처:[^\n]*\]|출처:\s*\[[^\n]*\]\([^\n]+\))\s*)\n(?=\s*\d+\.\s+)",
        r"\1\n\n\n",
        text,
    )
    payload["answer"] = text
    payload["message"] = text
    payload["response"] = text


_SOURCE_BLOCK_RE = re.compile(
    r"\n(?:---\s*\n)?(?:\*\*원본 PDF\*\*|원본 PDF)\s*\n[\s\S]*$",
    flags=re.IGNORECASE,
)


def _strip_existing_source_pdf_block(text: str) -> str:
    """Remove any model-generated source section so server appends a single canonical block."""
    t = str(text or "")
    if not t:
        return t
    cleaned = _SOURCE_BLOCK_RE.sub("", t).rstrip()
    return cleaned


_SOURCE_PDF_APPENDED_MARKER = "\n\n---\n**원본 PDF**\n"


def _markdown_bullets_from_links(links: List[Any]) -> List[str]:
    lines: List[str] = []
    for x in links:
        if not isinstance(x, dict):
            continue
        url = x.get("url")
        if not url:
            continue
        raw_label = str(x.get("label") or "PDF")
        safe_label = _markdown_source_link_label(raw_label)
        lines.append(f"- [{safe_label}]({url})")
    return lines


def _plain_answer_body_enabled() -> bool:
    return (os.environ.get("GSLLM_PLAIN_ANSWER", "1") or "").strip() != "0"


def _markdown_body_to_plain(text: str) -> str:
    s = str(text or "")
    if not s.strip():
        return s
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    lines_out: List[str] = []
    for line in s.split("\n"):
        raw = line.rstrip("\r")
        stripped = raw.strip()
        if "|" in stripped and stripped.count("|") >= 2:
            if re.match(r"^[\s|:\.-]+$", stripped) and "-" in stripped:
                continue
            cells = [c.strip() for c in raw.split("|")]
            cells = [c for c in cells if c]
            if cells:
                lines_out.append(" · ".join(cells))
            continue
        ln = raw
        ln = re.sub(r"^#{1,6}\s+", "", ln)
        ln = re.sub(r"^\s*>+\s?", "", ln)
        lines_out.append(ln)
    s = "\n".join(lines_out)
    for _ in range(8):
        prev = s
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
        s = re.sub(r"__([^_]+)__", r"\1", s)
        if s == prev:
            break

    def _http_md_plain(m: re.Match[str]) -> str:
        lbl, url = m.group(1), m.group(2)
        if "/gsllm/source-pdf" in url:
            return m.group(0)
        return f"{lbl} ({url})"

    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", _http_md_plain, s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def _apply_plain_answer_keep_source_block(payload: Dict[str, Any]) -> None:
    if not _plain_answer_body_enabled():
        return
    full = str(payload.get("answer") or "")
    if not full:
        return
    marker = _SOURCE_PDF_APPENDED_MARKER
    if marker in full:
        head, tail = full.split(marker, 1)
        new_full = _markdown_body_to_plain(head).rstrip() + marker + tail.lstrip("\n")
    else:
        new_full = _markdown_body_to_plain(full)
    payload["answer"] = new_full
    payload["message"] = new_full
    payload["response"] = new_full


def _append_answer_source_links_markdown(payload: Dict[str, Any]) -> None:
    if (os.environ.get("GSLLM_APPEND_SOURCE_LINKS", "1") or "").strip() == "0":
        return
    mode = _source_links_mode()
    retrieval = [x for x in (payload.get("source_page_links_retrieval") or []) if isinstance(x, dict)]
    cited = [x for x in (payload.get("source_page_links_cited") or []) if isinstance(x, dict)]
    if mode == "retrieval":
        links = [x for x in (payload.get("source_page_links") or []) if isinstance(x, dict)] or retrieval
    elif mode == "cited":
        links = [x for x in (payload.get("source_page_links") or []) if isinstance(x, dict)] or cited
    else:
        links = []

    lines: List[str] = []
    if mode == "both":
        c_lines = _markdown_bullets_from_links(cited)
        r_lines = _markdown_bullets_from_links(retrieval)
        inner_parts: List[str] = []
        if c_lines:
            inner_parts.append("**답변 근거**\n" + "\n".join(c_lines))
        if r_lines:
            inner_parts.append("**검색 참고 문서**\n" + "\n".join(r_lines))
        if not inner_parts:
            return
        block = _SOURCE_PDF_APPENDED_MARKER + "\n\n".join(inner_parts)
    else:
        lines = _markdown_bullets_from_links(links)
        if (
            not lines
            and mode == "cited"
            and not bool(payload.get("with_answer"))
            and retrieval
        ):
            lines = _markdown_bullets_from_links(retrieval)
        if not lines:
            return
        block = _SOURCE_PDF_APPENDED_MARKER + "\n".join(lines)

    base_ans = _strip_existing_source_pdf_block(str(payload.get("answer") or ""))
    new_ans = base_ans + block
    payload["answer"] = new_ans
    payload["message"] = new_ans
    payload["response"] = new_ans


def _warm_query_models() -> None:
    global _query_embedder, _query_reranker, _query_device, _query_embed_id, _query_reranker_id
    device = embed_query_all.pick_device(os.environ.get("GSLLM_QUERY_DEVICE"))
    embed_id = os.environ.get("GSLLM_EMBED_MODEL", embed_query_all.EMBED_MODEL)
    reranker_id = os.environ.get("GSLLM_RERANKER_MODEL", embed_query_all.RERANKER_NAME)
    from sentence_transformers import CrossEncoder, SentenceTransformer

    _query_embedder = SentenceTransformer(embed_id, device=device)
    _query_reranker = CrossEncoder(reranker_id, device=device)
    _query_device = device
    _query_embed_id = embed_id
    _query_reranker_id = reranker_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_query_models()
    yield


app = FastAPI(title="gsllm-api", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GsllmEmbedRequest(BaseModel):
    filepath: str = Field(
        ...,
        description="단일 문서 절대 경로: .pdf 또는 .doc/.docx/.xls/.xlsx/.ppt/.pptx 등 (Office는 PDF 변환 후 임베딩)",
    )
    work_dir: Optional[str] = None
    chroma_dir: Optional[str] = None
    collection: str = embed_docs.DEFAULT_COLLECTION
    embed_model: Optional[str] = None
    enable_ocr: bool = True
    ocr_lang: str = "kor+eng"
    save_page_images: bool = False
    verbose: bool = False
    email: Optional[str] = None
    role: Optional[str] = None
    original_filename: Optional[str] = Field(
        None,
        description="업로드 시 사용자 파일명(청크 metadata.original_filename). 생략 시 서버 저장 파일명 사용.",
    )


class GsllmQueryRequest(BaseModel):
    query: str
    chroma_base: Optional[str] = None
    per_collection_k: int = 20
    final_k: int = 10
    excerpt_chars: int = embed_query_all.DOC_EXCERPT_CHARS
    collection: Optional[str] = None
    with_answer: bool = True
    model_id: Optional[str] = f"local:{embed_query_all.DEFAULT_LOCAL_LLM_MODEL}"
    llm_n_ctx: int = embed_query_all.DEFAULT_LLM_N_CTX
    llm_max_tokens: int = embed_query_all.DEFAULT_LLM_MAX_TOKENS
    email: Optional[str] = None
    role: Optional[str] = None


class GsllmDocumentDeleteRequest(BaseModel):
    """GET /gsllm/documents 의 storage_basename 또는 doc_key 로 논리 문서 1건 삭제."""

    email: str
    role: str
    chroma_dir: Optional[str] = None
    store: Optional[str] = None
    collection: Optional[str] = None
    doc_key: Optional[str] = Field(
        None,
        description="논리 문서 UUID 문자열(GET documents 의 doc_key). 있으면 Chroma 에서 해당 키로만 삭제(빠름).",
    )
    storage_basename: Optional[str] = None
    filename: Optional[str] = Field(
        None,
        description="doc_key 없을 때: storage_basename 과 택일. 디스크/메타 파일명 권장.",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "gsllm-api"}


@app.get("/gsllm/info")
def info() -> dict[str, Any]:
    bases = [str(p) for p in _allow_prefixes()]
    return {
        "path_allow_prefixes": bases,
        "defaults": {
            "chroma_base": str(_default_chroma_parent()),
            "data_base": str(_default_data_parent()),
            "work_base": str(_default_work_parent()),
        },
        "query_models": {
            "device": _query_device,
            "embed_model": _query_embed_id,
            "reranker": _query_reranker_id,
        },
    }


def _sanitize_source_pdf_display_name(raw: str, *, max_len: int = 200) -> str:
    s = unquote((raw or "").strip())[:max_len]
    s = s.replace("\r", "").replace("\n", "").replace('"', "'")
    for ch in (";", "\x00"):
        s = s.replace(ch, "")
    return s.strip() or ""


@app.get("/gsllm/source-pdf")
def gsllm_source_pdf(
    email: str = Query(..., description="세션과 동일한 사용자 이메일"),
    role: str = Query(..., description="세션과 동일한 role"),
    store: str = Query(..., description="data 하위 스토어 폴더명(예: collection_A)"),
    file: str = Query(..., description="PDF 파일 베이스명만 (경로 금지)"),
    as_: Optional[str] = Query(
        None,
        alias="as",
        description="저장/표시용 원본 파일명(선택, UTF-8). Content-Disposition filename에 반영",
    ),
) -> FileResponse:
    """
    임베딩된 원본 PDF를 인라인으로 제공. 브라우저/뷰어는 URL fragment `#page=N`으로 페이지 이동.
    """
    _, role_n, eff_store = _enforce_collection_access(
        email=email,
        role=role,
        requested_collection=store.strip(),
        endpoint="GET /gsllm/source-pdf",
        require_collection=True,
    )
    store_n = str(eff_store or store).strip()
    if not store_n:
        raise HTTPException(status_code=400, detail="invalid store")

    raw_name = (file or "").strip()
    if not raw_name or raw_name != Path(raw_name).name:
        raise HTTPException(status_code=400, detail="invalid file parameter")
    if "/" in raw_name or "\\" in raw_name or raw_name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid file parameter")

    basename = Path(raw_name).name
    if not basename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only .pdf is supported")

    data_root = _default_data_parent()
    full = (data_root / store_n / basename).resolve()
    bases = _allow_prefixes()
    if not _is_allowed_path(full, bases):
        raise HTTPException(status_code=400, detail="path not allowed")
    if not full.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if full.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="not a pdf file")

    disp = _sanitize_source_pdf_display_name(as_ or "")
    if disp and not disp.lower().endswith(".pdf"):
        disp = f"{disp}.pdf"
    filename_hdr = disp or basename

    return FileResponse(
        path=str(full),
        media_type="application/pdf",
        filename=filename_hdr,
        content_disposition_type="inline",
    )


# --- emb.py 호환: siwasoftwebtest `rag-collections.js` 가 8010만 쓸 때 동일 경로로 연동 ---


def _discover_multi_store_names(chroma_parent: Path, data_parent: Path) -> List[str]:
    """
    스토어 폴더 후보 이름: chroma 부모 아래 디렉터리 + data 부모 아래 디렉터리(숨김 제외) 합집합.
    chroma.sqlite3 가 아직 없어도(빈 스토어·정리 직후) 웹 드롭다운에 나오게 한다.
    """
    names: set[str] = set()
    if chroma_parent.is_dir():
        for c in chroma_parent.iterdir():
            if c.is_dir() and not c.name.startswith("."):
                names.add(c.name)
    dp = data_parent.resolve()
    cp = chroma_parent.resolve()
    if data_parent.is_dir() and dp != cp:
        for c in data_parent.iterdir():
            if c.is_dir() and not c.name.startswith("."):
                names.add(c.name)
    return sorted(names)


@app.get("/collections")
def list_collections(
    chroma: Optional[str] = None,
    email: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
) -> dict[str, Any]:
    _, role_n, _ = _enforce_collection_access(
        email=email,
        role=role,
        requested_collection=None,
        endpoint="GET /collections",
    )
    allowed = _allowed_collections(role_n)
    bases = _allow_prefixes()
    base = _resolve_chroma_arg(chroma)
    if not _is_allowed_path(base, bases):
        raise HTTPException(status_code=400, detail=f"chroma path not allowed: {base}")

    if (base / "chroma.sqlite3").is_file():
        try:
            out = _run_chroma_subprocess("list", path=base)
            out_names = sorted(list(out.get("collections") or []))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"failed to list collections at {base}: {e}",
            ) from e
        if allowed is not None:
            out_names = [n for n in out_names if n in allowed]
        return {"ok": True, "chroma": str(base), "collections": out_names, "effective_role": role_n}

    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {base}")

    store_names = _discover_multi_store_names(base, _default_data_parent())
    if allowed is not None:
        store_names = [n for n in store_names if n in allowed]
    return {"ok": True, "chroma": str(base), "collections": store_names, "effective_role": role_n}


@app.post("/collections")
def create_collection(body: Dict[str, Any]) -> dict[str, Any]:
    name = body.get("name")
    if not name or not str(name).strip():
        raise HTTPException(status_code=400, detail="name is required")
    name = str(name).strip()
    _, role_n, eff_name = _enforce_collection_access(
        email=body.get("email"),
        role=body.get("role"),
        requested_collection=name,
        endpoint="POST /collections",
        require_collection=True,
    )
    name = str(eff_name)
    bases = _allow_prefixes()
    parent = _resolve_chroma_arg(body.get("chroma"))
    if not _is_allowed_path(parent, bases):
        raise HTTPException(status_code=400, detail=f"chroma path not allowed: {parent}")
    metadata = body.get("metadata") or {"hnsw:space": "cosine"}
    lock = _collection_lock(name)

    with lock:
        _wait_if_recently_deleted(name)
        if (parent / "chroma.sqlite3").is_file():
            try:
                _run_chroma_subprocess(
                    "create",
                    path=parent,
                    name=name,
                    metadata=metadata,
                )
            except Exception as e:
                print(f"[CREATE_COLLECTION_ERROR] name={name} parent={parent} error={repr(e)}")
                traceback.print_exc(limit=20)
                if _is_chroma_readonly_error(e):
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"failed to initialize chroma collection at {parent}: {e}. "
                            "서버 로그의 [CREATE_COLLECTION_ERROR]를 확인하세요."
                        ),
                    ) from e
                raise HTTPException(status_code=500, detail=f"failed to create collection at {parent}: {e}") from e
            return {"ok": True, "name": name, "chroma": str(parent), "effective_collection": name, "effective_role": role_n}

        if not parent.is_dir():
            parent.mkdir(parents=True, exist_ok=True)

        store_dir = (parent / name).resolve()
        if not _is_allowed_path(store_dir, bases):
            raise HTTPException(status_code=400, detail=f"store path not allowed: {store_dir}")
        created_store_dir = not store_dir.exists()
        store_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run_chroma_subprocess(
                "create",
                path=store_dir,
                name=name,
                metadata=metadata,
            )
        except Exception as e:
            print(f"[CREATE_COLLECTION_ERROR] name={name} store_dir={store_dir} error={repr(e)}")
            traceback.print_exc(limit=20)
            if created_store_dir:
                shutil.rmtree(store_dir, ignore_errors=True)
            if _is_chroma_readonly_error(e):
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"failed to initialize chroma collection at {store_dir}: {e}. "
                        "서버 로그의 [CREATE_COLLECTION_ERROR]를 확인하세요."
                    ),
                ) from e
            raise HTTPException(status_code=500, detail=f"failed to create collection at {store_dir}: {e}") from e

        # 요청한 흐름 지원: 컬렉션 생성 시 data/<collection> 스테이징 폴더도 함께 생성
        data_parent = _default_data_parent()
        data_dir = (data_parent / name).resolve()
        if _is_allowed_path(data_dir, bases):
            data_dir.mkdir(parents=True, exist_ok=True)

        return {
            "ok": True,
            "name": name,
            "chroma": str(store_dir),
            "data_dir": str(data_dir),
            "effective_collection": name,
            "effective_role": role_n,
        }


@app.delete("/collections")
def delete_collection(
    name: str = Query(..., description="컬렉션(또는 스토어 폴더) 이름"),
    chroma: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
) -> dict[str, Any]:
    if not name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    _, role_n, eff_name = _enforce_collection_access(
        email=email,
        role=role,
        requested_collection=name.strip(),
        endpoint="DELETE /collections",
        require_collection=True,
    )
    name = str(eff_name)
    bases = _allow_prefixes()
    parent = _resolve_chroma_arg(chroma)
    if not _is_allowed_path(parent, bases):
        raise HTTPException(status_code=400, detail=f"chroma path not allowed: {parent}")
    data_parent = _default_data_parent()
    work_parent = _default_work_parent()
    data_dir = (data_parent / name).resolve()
    work_dir = (work_parent / name).resolve()
    lock = _collection_lock(name)

    with lock:
        if (parent / "chroma.sqlite3").is_file():
            try:
                _run_chroma_subprocess(
                    "delete_collection",
                    path=parent,
                    name=name,
                )
            except Exception as e:
                print(f"[DELETE_COLLECTION_ERROR] name={name} parent={parent} error={repr(e)}")
                traceback.print_exc(limit=20)
                raise HTTPException(
                    status_code=500,
                    detail=f"failed to delete chroma collection at {parent}: {e}",
                ) from e
            time.sleep(0.5)
            deleted_paths: Dict[str, bool] = {"chroma_collection_only": True}
            for label, target in (("data_dir", data_dir), ("work_dir", work_dir)):
                if not _is_allowed_path(target, bases):
                    deleted_paths[label] = False
                    continue
                deleted_paths[label] = _safe_rmtree(target, label=label) if target.is_dir() else False
            _mark_collection_deleted(name)
            return {
                "ok": True,
                "deleted": name,
                "chroma": str(parent),
                "effective_collection": name,
                "effective_role": role_n,
                "deleted_paths": deleted_paths,
            }

        store_dir = (parent / name).resolve()
        if not store_dir.is_dir() or not (store_dir / "chroma.sqlite3").is_file():
            raise HTTPException(status_code=404, detail=f"store not found: {store_dir}")
        if not _is_allowed_path(store_dir, bases):
            raise HTTPException(status_code=400, detail="store path not allowed")
        try:
            _run_chroma_subprocess("delete_store", path=store_dir)
        except Exception as e:
            print(f"[DELETE_COLLECTION_ERROR] name={name} store_dir={store_dir} error={repr(e)}")
            traceback.print_exc(limit=20)
            raise HTTPException(
                status_code=500,
                detail=f"failed to delete chroma store via subprocess: {store_dir} ({e})",
            ) from e
        time.sleep(0.5)
        deleted_paths: Dict[str, bool] = {
            "chroma_dir": not store_dir.exists(),
            "data_dir": False,
            "work_dir": False,
        }
        for label, target in (("data_dir", data_dir), ("work_dir", work_dir)):
            if not _is_allowed_path(target, bases):
                deleted_paths[label] = False
                continue
            deleted_paths[label] = _safe_rmtree(target, label=label) if target.is_dir() else False
        gc.collect()
        _mark_collection_deleted(name)
        return {
            "ok": True,
            "deleted": name,
            "chroma": str(store_dir),
            "effective_collection": name,
            "effective_role": role_n,
            "deleted_paths": deleted_paths,
        }


def _execute_embed_request(
    body: GsllmEmbedRequest,
    *,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    if progress_cb:
        progress_cb({"stage": "validating", "message": "요청 검증 중"})
    bases = _allow_prefixes()
    src_path = Path(body.filepath).expanduser()
    if not src_path.is_file():
        raise HTTPException(status_code=400, detail="filepath must be an existing file")

    ext = src_path.suffix.lower()
    if ext != ".pdf" and ext not in _OFFICE_TO_PDF_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: {ext}. Use .pdf or Office formats: {sorted(_OFFICE_TO_PDF_EXT)}",
        )

    _, role_n, effective_coll = _enforce_collection_access(
        email=body.email,
        role=body.role,
        requested_collection=(body.collection or "").strip() or None,
        endpoint="POST /gsllm/embed",
    )
    coll = str(effective_coll or embed_docs.DEFAULT_COLLECTION).strip()
    if body.work_dir and str(body.work_dir).strip():
        work_dir = Path(body.work_dir).expanduser()
    elif os.environ.get("GSLLM_EMBED_WORK_DIR"):
        work_dir = Path(os.environ["GSLLM_EMBED_WORK_DIR"]).expanduser()
    else:
        work_root = Path(embed_docs.DEFAULT_WORK_DIR).parent
        work_dir = (work_root / coll).expanduser()

    if body.chroma_dir and str(body.chroma_dir).strip():
        chroma_dir = Path(body.chroma_dir).expanduser()
    elif os.environ.get("GSLLM_EMBED_CHROMA_DIR"):
        chroma_dir = Path(os.environ["GSLLM_EMBED_CHROMA_DIR"]).expanduser()
    else:
        chroma_dir = (_default_chroma_parent() / coll).expanduser()

    embed_model = body.embed_model or embed_docs.DEFAULT_EMBED_MODEL

    for label, p in (("filepath", src_path), ("work_dir", work_dir), ("chroma_dir", chroma_dir)):
        if not _is_allowed_path(p, bases):
            raise HTTPException(
                status_code=400,
                detail=f"{label} is outside GSLLM_PATH_ALLOW_PREFIX: {p}",
            )

    converted_from: Optional[str] = None
    pdf_path = src_path
    if ext in _OFFICE_TO_PDF_EXT:
        if progress_cb:
            progress_cb({"stage": "converting", "message": "Office 문서를 PDF로 변환 중"})
        try:
            timeout_sec = int(os.environ.get("GSLLM_CONVERT_TIMEOUT_SEC", "300"))
        except ValueError:
            timeout_sec = 300
        try:
            pdf_path = _convert_office_to_pdf(src_path, timeout_sec=timeout_sec)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        converted_from = str(src_path.resolve())
        if not _is_allowed_path(pdf_path, bases):
            raise HTTPException(
                status_code=400,
                detail=f"converted pdf is outside GSLLM_PATH_ALLOW_PREFIX: {pdf_path}",
            )

    orig_fn = (body.original_filename or "").strip() or src_path.name

    chroma_resolved = chroma_dir.resolve()
    c_lock = _chroma_persist_lock(chroma_resolved)
    ns = argparse.Namespace(
        inputs=[str(pdf_path.resolve())],
        recursive=False,
        work_dir=str(work_dir.resolve()),
        chroma_dir=str(chroma_resolved),
        collection=coll,
        embed_model=embed_model,
        enable_ocr=body.enable_ocr,
        ocr_lang=body.ocr_lang,
        save_page_images=body.save_page_images,
        verbose=body.verbose,
        original_filename=orig_fn,
        chroma_lock=c_lock,
    )

    # 동일 original_filename purge 는 embed_docs.run_embed_job 안에서 upsert 와 같은 chroma_lock 구간에서 수행

    if progress_cb:
        progress_cb({"stage": "embedding", "message": "문서 임베딩 작업 실행 중"})
    embed_docs.configure_logging(ns.verbose)
    try:
        code, summary = embed_docs.run_embed_job(ns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if code != 0:
        raise HTTPException(status_code=400, detail=summary)

    elapsed = time.perf_counter() - t0
    end_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    # 웹 경유 호출에서도 embed_docs.py CLI와 동일하게 터미널 마지막에 소요 시간을 남긴다.
    print(
        f"[TIME] 시작 {start_wall} → 종료 {end_wall} | 총 소요 "
        f"{embed_docs._format_run_duration(elapsed)}"
    )
    summary["elapsed_sec"] = round(elapsed, 3)
    summary["embed_pdf_path"] = str(pdf_path.resolve())
    summary["source_filepath"] = str(src_path.resolve())
    summary["effective_collection"] = coll
    summary["effective_role"] = role_n
    summary["original_filename"] = orig_fn
    if converted_from:
        summary["converted_from_office"] = converted_from
        summary["converted_pdf_path"] = str(pdf_path.resolve())
    if progress_cb:
        progress_cb({"stage": "finalizing", "message": "결과 정리 중"})
    return summary


def _set_embed_job_state(job_id: str, **fields: Any) -> None:
    with _EMBED_JOBS_LOCK:
        rec = _EMBED_JOBS.get(job_id)
        if not rec:
            return
        rec.update(fields)
        rec["updated_at"] = _now_iso()
        rec["updated_ts"] = time.time()


def _run_embed_job_worker(job_id: str, body_payload: Dict[str, Any]) -> None:
    def _progress(p: Dict[str, Any]) -> None:
        _set_embed_job_state(job_id, progress=p)

    _set_embed_job_state(job_id, status="running", progress={"stage": "starting", "message": "작업 시작"})
    try:
        body = GsllmEmbedRequest(**body_payload)
        result = _execute_embed_request(body, progress_cb=_progress)
    except HTTPException as e:
        err = {
            "code": f"http_{e.status_code}",
            "message": str(e.detail),
        }
        _set_embed_job_state(job_id, status="failed", error=err)
        return
    except Exception as e:
        err = {
            "code": "internal_error",
            "message": str(e),
            "detail": traceback.format_exc(limit=20),
        }
        _set_embed_job_state(job_id, status="failed", error=err)
        return
    _set_embed_job_state(
        job_id,
        status="succeeded",
        result=result,
        progress={"stage": "done", "message": "임베딩 완료"},
    )


@app.post(
    "/gsllm/embed",
    tags=["embed"],
    summary="동기 임베딩(전체 작업이 끝날 때까지 응답 대기)",
    description=(
        "한 HTTP 요청 안에서 임베딩이 끝까지 실행된다. 프록시·브라우저 타임아웃과 맞지 않기 쉬우므로 "
        "웹 앱 기본 경로로는 POST /gsllm/embed-jobs 를 사용한다. curl·내부 도구용으로 유지."
    ),
)
def gsllm_embed(body: GsllmEmbedRequest) -> dict[str, Any]:
    return _execute_embed_request(body)


@app.post(
    "/gsllm/embed-jobs",
    tags=["embed"],
    summary="비동기 임베딩 job 등록",
    description=(
        "요청 본문 검증 후 즉시 job_id 를 반환하고, 실제 임베딩은 백그라운드 스레드에서 실행된다. "
        "진행 상황은 GET /gsllm/embed-jobs/{job_id} 로 폴링한다."
    ),
)
@app.post(
    "/embed-jobs",
    tags=["embed"],
    summary="비동기 임베딩 job 등록(별칭)",
    description="POST /gsllm/embed-jobs 와 동일.",
)
def create_embed_job(body: GsllmEmbedRequest) -> dict[str, Any]:
    # 본문 파싱은 FastAPI가 핸들러 진입 전에 수행(대용량이면 그만큼 지연). 이하는 prune·dict 삽입·스레드 시작만 동기.
    _prune_embed_jobs()
    job_id = str(uuid.uuid4())
    now_iso = _now_iso()
    now_ts = time.time()
    rec: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now_iso,
        "updated_at": now_iso,
        "created_ts": now_ts,
        "updated_ts": now_ts,
        "progress": {"stage": "queued", "message": "작업 대기 중"},
        "request": body.dict(),
        "result": None,
        "error": None,
    }
    with _EMBED_JOBS_LOCK:
        _EMBED_JOBS[job_id] = rec
    t = threading.Thread(
        target=_run_embed_job_worker,
        args=(job_id, body.dict()),
        name=f"embed-job-{job_id[:8]}",
        daemon=True,
    )
    t.start()
    return {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "created_at": now_iso,
    }


@app.get(
    "/gsllm/embed-jobs/{job_id}",
    tags=["embed"],
    summary="임베딩 job 상태 조회",
    description=(
        "응답의 `status` 로 queued|running|succeeded|failed|cancelled 를 구분한다(비정상 레코드는 unknown). "
        "진행 중에는 `ok` 가 false 일 수 있으므로 성공 여부는 `status == succeeded` 로만 판단한다."
    ),
)
@app.get(
    "/embed-jobs/{job_id}",
    tags=["embed"],
    summary="임베딩 job 상태 조회(별칭)",
    description="GET /gsllm/embed-jobs/{job_id} 와 동일.",
)
def get_embed_job(job_id: str) -> dict[str, Any]:
    _prune_embed_jobs()
    with _EMBED_JOBS_LOCK:
        rec = _EMBED_JOBS.get(job_id)
        if not rec:
            raise HTTPException(status_code=404, detail=f"embed job not found: {job_id}")
        out = dict(rec)
    return _job_public_view(out)


def _storage_basename_from_chunk_meta(md: Dict[str, Any]) -> str:
    fp = md.get("file_path") or md.get("source_path") or ""
    if fp:
        return Path(str(fp)).name
    fn = md.get("filename") or md.get("file_name") or ""
    return Path(str(fn)).name if str(fn).strip() else ""


def _resolve_gsllm_documents_chroma_path(
    *,
    chroma_dir: Optional[str],
    store: Optional[str],
    collection: Optional[str],
    bases: List[Path],
) -> tuple[Path, Optional[str], str]:
    p: Optional[Path] = None
    coll_name: Optional[str] = (collection or "").strip() or None
    st = (store or "").strip() if store else ""

    if chroma_dir and str(chroma_dir).strip():
        p = Path(str(chroma_dir).strip()).expanduser().resolve()
    elif st:
        p = (_default_chroma_parent() / st).resolve()
        if not coll_name:
            coll_name = st
    else:
        raise HTTPException(status_code=400, detail="chroma_dir 또는 store 쿼리가 필요합니다.")

    if not _is_allowed_path(p, bases):
        raise HTTPException(status_code=400, detail=f"path not allowed: {p}")
    if not p.is_dir() or not (p / "chroma.sqlite3").is_file():
        raise HTTPException(status_code=400, detail=f"Chroma 데이터 디렉터리가 아닙니다: {p}")

    rbac_store = st or p.name
    return p, coll_name, rbac_store


def _normalize_document_delete_target(storage_basename: Optional[str], filename: Optional[str]) -> str:
    raw = (storage_basename or filename or "").strip()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="storage_basename 또는 filename이 필요합니다.",
        )
    base = Path(raw).name
    if not base or base in (".", ".."):
        raise HTTPException(status_code=400, detail="잘못된 문서 식별자입니다.")
    return base


def _chunk_meta_matches_delete_target(md: Dict[str, Any], target: str) -> bool:
    md = dict(md or {})
    sb = _storage_basename_from_chunk_meta(md)
    if sb == target:
        return True
    orig = str(md.get("original_filename") or "").strip()
    if orig == target:
        return True
    for key in ("filename", "file_name"):
        fn = str(md.get(key) or "").strip()
        if fn and (fn == target or Path(fn).name == target):
            return True
    for key in ("file_path", "source_path"):
        fp = md.get(key) or ""
        if fp and Path(str(fp)).name == target:
            return True
    return False


@app.get("/gsllm/documents")
def gsllm_documents(
    email: str = Query(..., description="세션과 동일한 사용자 이메일"),
    role: str = Query(..., description="세션과 동일한 role"),
    chroma_dir: Optional[str] = Query(
        None,
        description="Chroma PersistentClient 경로 (…/embed_test/chroma/collection_A)",
    ),
    store: Optional[str] = Query(
        None,
        description="스토어 폴더명만 넣을 때 (부모는 GSLLM_CHROMA_BASE)",
    ),
    collection: Optional[str] = Query(
        None,
        description="컬렉션명; 생략 시 store명 또는 DB 내 단일 컬렉션",
    ),
) -> dict[str, Any]:
    """
    설정 화면 «임베딩된 문서 목록»용. 청크 ID는 대표 1개만, 파일 단위로 묶어 반환.
    siwasoftwebtest `rag-documents` 프록시가 이 경로를 쓰도록 연동 가능.
    """
    bases = _allow_prefixes()
    p, coll_name, rbac_store = _resolve_gsllm_documents_chroma_path(
        chroma_dir=chroma_dir,
        store=store,
        collection=collection,
        bases=bases,
    )
    _, role_n, eff_coll = _enforce_collection_access(
        email=email,
        role=role,
        requested_collection=rbac_store,
        endpoint="GET /gsllm/documents",
        require_collection=True,
    )

    c_lock = _chroma_persist_lock(p)
    client: Any = None
    with c_lock:
        try:
            try:
                client = _persistent_chroma_client(p)
                col_names = sorted([c.name for c in client.list_collections()])
            except ValueError as e:
                if "different settings" in str(e).lower():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Chroma 클라이언트 설정 충돌입니다. 임베딩 작업이 같은 DB를 붙잡고 있으면 끝난 뒤 다시 시도하세요. "
                            f"{e}"
                        ),
                    ) from e
                raise HTTPException(status_code=500, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Chroma 열기 실패: {e}") from e
            if not col_names:
                return {"ok": True, "chroma_dir": str(p), "documents": []}

            if coll_name:
                if coll_name not in col_names:
                    raise HTTPException(
                        status_code=404, detail=f"컬렉션 없음: {coll_name}. 사용 가능: {col_names}"
                    )
                use_name = coll_name
            elif len(col_names) == 1:
                use_name = col_names[0]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"collection 쿼리로 이름을 지정하세요. 후보: {col_names}",
                )

            try:
                col = client.get_collection(use_name)
                raw = col.get(include=["metadatas", "documents"])
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Chroma 조회 실패(잠시 후 재시도): {e}",
                ) from e
            ids = raw.get("ids") or []
            metas = raw.get("metadatas") or []
            docs = raw.get("documents") or []

            by_file: Dict[str, Dict[str, Any]] = {}
            for i, doc_id in enumerate(ids):
                md = dict(metas[i] or {})
                doc = docs[i] if i < len(docs) else ""
                sb = _storage_basename_from_chunk_meta(md) or f"document_{i+1}"
                dk_meta = str(md.get("doc_key") or "").strip()
                orig = str(md.get("original_filename") or "").strip()
                display = orig or sb
                size_add = len(doc) if isinstance(doc, str) else 0
                if sb not in by_file:
                    created_at = str(md.get("created_at", "") or md.get("embedded_at", "") or "")
                    by_file[sb] = {
                        "id": doc_id,
                        "storage_basename": sb,
                        "original_filename": orig or None,
                        "display_name": display,
                        "filename": display,
                        "doc_key": dk_meta or None,
                        "created_at": created_at,
                        "size": size_add,
                        "chunk_count": 1,
                        "type": "embedded",
                    }
                else:
                    entry = by_file[sb]
                    entry["size"] = int(entry.get("size") or 0) + size_add
                    entry["chunk_count"] = int(entry.get("chunk_count") or 0) + 1
                    if orig:
                        entry["original_filename"] = orig
                        entry["display_name"] = orig
                        entry["filename"] = orig
                    if dk_meta and not entry.get("doc_key"):
                        entry["doc_key"] = dk_meta

            documents = sorted(by_file.values(), key=lambda x: str(x.get("display_name") or x.get("filename", "")))
            return {
                "ok": True,
                "chroma_dir": str(p),
                "collection": use_name,
                "documents": documents,
                "effective_role": role_n,
                "effective_collection": str(eff_coll or rbac_store),
            }
        finally:
            _close_chroma_client(client)


def _disk_basename_from_storage_or_filename(storage_basename: Optional[str], filename: Optional[str]) -> str:
    raw = str(storage_basename or filename or "").strip()
    if not raw:
        return ""
    bn = Path(raw).name
    return bn if bn not in ("", ".", "..") else ""


def _run_gsllm_document_embed_delete(
    *,
    email: str,
    role: str,
    chroma_dir: Optional[str],
    store: Optional[str],
    collection: Optional[str],
    storage_basename: Optional[str],
    filename: Optional[str],
    doc_key: Optional[str] = None,
) -> dict[str, Any]:
    bases = _allow_prefixes()
    dk = str(doc_key or "").strip()
    explicit_disk_bn = _disk_basename_from_storage_or_filename(storage_basename, filename)
    target_basename = ""
    if not dk:
        target_basename = _normalize_document_delete_target(storage_basename, filename)

    p, coll_name, rbac_store = _resolve_gsllm_documents_chroma_path(
        chroma_dir=chroma_dir,
        store=store,
        collection=collection,
        bases=bases,
    )
    _, role_n, eff_coll = _enforce_collection_access(
        email=email,
        role=role,
        requested_collection=rbac_store,
        endpoint="DELETE /gsllm/documents/embed",
        require_collection=True,
    )

    deleted_chunk_count = 0
    disk_bn = ""
    c_lock = _chroma_persist_lock(p)
    client: Any = None
    use_name = ""

    with c_lock:
        try:
            try:
                client = _persistent_chroma_client(p)
                col_names = sorted([c.name for c in client.list_collections()])
            except ValueError as e:
                if "different settings" in str(e).lower():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Chroma 클라이언트 설정 충돌입니다. 임베딩 작업이 같은 DB를 붙잡고 있으면 끝난 뒤 다시 시도하세요. "
                            f"{e}"
                        ),
                    ) from e
                raise HTTPException(status_code=500, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Chroma 열기 실패: {e}") from e

            if not col_names:
                placeholder = dk or explicit_disk_bn or target_basename
                return {
                    "ok": True,
                    "deleted_chunk_count": 0,
                    "deleted_data_path": None,
                    "deleted_work_paths": [],
                    "store": rbac_store,
                    "effective_collection": str(eff_coll or rbac_store),
                    "effective_role": role_n,
                    "chroma_dir": str(p),
                    "collection": None,
                    "target": placeholder,
                    "doc_key": dk or None,
                    "note": "persist 디렉터리에 Chroma 컬렉션이 없습니다.",
                }

            if coll_name:
                if coll_name not in col_names:
                    raise HTTPException(
                        status_code=404,
                        detail=f"컬렉션 없음: {coll_name}. 사용 가능: {col_names}",
                    )
                use_name = coll_name
            elif len(col_names) == 1:
                use_name = col_names[0]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"collection 쿼리로 이름을 지정하세요. 후보: {col_names}",
                )

            col = client.get_collection(use_name)

            if dk:
                got = col.get(where={"doc_key": {"$eq": dk}}, include=["metadatas"])
                met_sample = got.get("metadatas") or []
                deleted_chunk_count = len(got.get("ids") or [])
                inferred_bn = ""
                if met_sample:
                    sb_meta = _storage_basename_from_chunk_meta(dict(met_sample[0] or {}))
                    inferred_bn = Path(str(sb_meta or "").strip()).name
                    if inferred_bn in ("", ".", ".."):
                        inferred_bn = ""
                try:
                    col.delete(where={"doc_key": {"$eq": dk}})
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Chroma delete(doc_key) 실패: {e}") from e
                disk_bn = inferred_bn or explicit_disk_bn
            else:
                raw = col.get(include=["metadatas"])
                ids_all = raw.get("ids") or []
                metas_all = raw.get("metadatas") or []
                to_delete_ids: List[str] = []
                for i, doc_id in enumerate(ids_all):
                    if _chunk_meta_matches_delete_target(dict(metas_all[i] or {}), target_basename):
                        to_delete_ids.append(str(doc_id))
                deleted_chunk_count = len(to_delete_ids)
                if to_delete_ids:
                    batch_sz = 500
                    for s in range(0, len(to_delete_ids), batch_sz):
                        col.delete(ids=to_delete_ids[s : s + batch_sz])
                disk_bn = target_basename
        finally:
            _close_chroma_client(client)

    deleted_data_path: Optional[str] = None
    if disk_bn:
        data_path = (_default_data_parent() / rbac_store / disk_bn).resolve()
        if data_path.is_file():
            if not _is_allowed_path(data_path, bases):
                raise HTTPException(status_code=400, detail=f"data path not allowed: {data_path}")
            try:
                data_path.unlink()
                deleted_data_path = str(data_path)
            except OSError as e:
                raise HTTPException(status_code=500, detail=f"data 파일 삭제 실패: {e}") from e

    deleted_work_paths: List[str] = []
    if disk_bn:
        work_col = (_default_work_parent() / rbac_store).resolve()
        page_img = (work_col / "page_images" / Path(disk_bn).stem).resolve()
        if page_img.is_dir():
            if not _is_allowed_path(page_img, bases):
                raise HTTPException(status_code=400, detail=f"work path not allowed: {page_img}")
            try:
                shutil.rmtree(page_img, ignore_errors=True)
                deleted_work_paths.append(str(page_img))
            except OSError as e:
                raise HTTPException(status_code=500, detail=f"work 디렉터리 삭제 실패: {e}") from e

    reply_target = disk_bn or dk or target_basename
    return {
        "ok": True,
        "deleted_chunk_count": deleted_chunk_count,
        "deleted_data_path": deleted_data_path,
        "deleted_work_paths": deleted_work_paths,
        "store": rbac_store,
        "effective_collection": str(eff_coll or rbac_store),
        "effective_role": role_n,
        "chroma_dir": str(p),
        "collection": use_name,
        "target": reply_target,
        "doc_key": dk or None,
    }


@app.delete(
    "/gsllm/documents/embed",
    tags=["documents"],
    summary="컬렉션 내 특정 문서 임베딩·원본 파일·page_images 제거",
)
def gsllm_documents_embed_delete(
    email: str = Query(..., description="세션과 동일한 사용자 이메일"),
    role: str = Query(..., description="세션과 동일한 role"),
    chroma_dir: Optional[str] = Query(
        None,
        description="Chroma PersistentClient 경로 (…/embed_test/chroma/collection_A)",
    ),
    store: Optional[str] = Query(None, description="스토어 폴더명 (부모는 GSLLM_CHROMA_BASE)"),
    collection: Optional[str] = Query(None, description="Chroma 논리 컬렉션명"),
    storage_basename: Optional[str] = Query(
        None,
        description="GET /gsllm/documents 목록의 storage_basename 과 동일",
    ),
    filename: Optional[str] = Query(
        None,
        description="storage_basename 과 택일 (디스크/메타 파일명)",
    ),
    doc_key: Optional[str] = Query(
        None,
        description="목록 문서 항목의 doc_key 와 동일(UUID). 있으면 Chroma 해당 키만 삭제(빠름)",
    ),
) -> dict[str, Any]:
    return _run_gsllm_document_embed_delete(
        email=email,
        role=role,
        chroma_dir=chroma_dir,
        store=store,
        collection=collection,
        storage_basename=storage_basename,
        filename=filename,
        doc_key=doc_key,
    )


@app.post(
    "/gsllm/documents/delete",
    tags=["documents"],
    summary="컬렉션 내 특정 문서 임베딩 제거(POST JSON)",
)
def gsllm_documents_delete_post(body: GsllmDocumentDeleteRequest) -> dict[str, Any]:
    return _run_gsllm_document_embed_delete(
        email=body.email,
        role=body.role,
        chroma_dir=body.chroma_dir,
        store=body.store,
        collection=body.collection,
        storage_basename=body.storage_basename,
        filename=body.filename,
        doc_key=body.doc_key,
    )


@app.post("/gsllm/query")
def gsllm_query(request: Request, body: GsllmQueryRequest) -> dict[str, Any]:
    if _query_embedder is None or _query_reranker is None:
        raise HTTPException(status_code=503, detail="Query models not initialized")

    email_n, role_n, effective_coll = _enforce_collection_access(
        email=body.email,
        role=body.role,
        requested_collection=body.collection,
        endpoint="POST /gsllm/query",
    )
    bases = _allow_prefixes()
    chroma_base = Path(body.chroma_base or embed_query_all.DEFAULT_CHROMA_BASE).expanduser()
    if not _is_allowed_path(chroma_base, bases):
        raise HTTPException(
            status_code=400,
            detail=f"chroma_base is outside GSLLM_PATH_ALLOW_PREFIX: {chroma_base}",
        )

    try:
        out = embed_query_all.run_single_query(
            body.query,
            chroma_base=chroma_base,
            embedder=_query_embedder,
            reranker=_query_reranker,
            per_collection_k=body.per_collection_k,
            final_k=body.final_k,
            excerpt_chars=body.excerpt_chars,
            collection_name=effective_coll,
            with_answer=body.with_answer,
            model_id=body.model_id,
            llm_n_ctx=body.llm_n_ctx,
            llm_max_tokens=body.llm_max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not out.get("ok"):
        raise HTTPException(status_code=400, detail=out)

    payload = dict(out)
    payload["effective_collection"] = effective_coll
    payload["effective_role"] = role_n
    ans = str(payload.get("answer") or "")
    payload["message"] = ans
    payload["response"] = ans

    public_base = _source_pdf_public_base(request)
    _enrich_source_pdf_links(
        payload,
        public_base=public_base,
        email=email_n,
        role=role_n,
    )
    payload["source_pdf_public_base"] = public_base
    _apply_citation_readability_payload(payload)
    _linkify_citations_in_answer(payload)
    _finalize_source_page_links_for_append(payload)
    _append_answer_source_links_markdown(payload)
    _apply_plain_answer_keep_source_block(payload)
    return payload


@app.post("/gsllm/chat")
def gsllm_chat_form(
    request: Request,
    query: str = Form(...),
    email: str = Form(...),
    role: str = Form(...),
) -> dict[str, Any]:
    """레거시 chatbot과 동일: `application/x-www-form-urlencoded`, 필드명 `query`."""
    return gsllm_query(request, GsllmQueryRequest(query=query, email=email, role=role))
