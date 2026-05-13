#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB 작업 전용 — FastAPI 프로세스와 분리해 한 번 실행 후 종료한다.
/collections 생성·삭제·목록에서 인프로세스 SharedSystemClient 캐시 이슈를 피하기 위함.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import chromadb


def create_collection(path: str, name: str, metadata_json: str) -> None:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)

    if metadata_json:
        metadata = json.loads(metadata_json)
    else:
        metadata = {"hnsw:space": "cosine"}

    client = chromadb.PersistentClient(path=str(p))
    client.get_or_create_collection(name=name, metadata=metadata)

    print(
        json.dumps(
            {
                "ok": True,
                "action": "create",
                "path": str(p),
                "name": name,
            },
            ensure_ascii=False,
        )
    )


def delete_store(path: str) -> None:
    p = Path(path).expanduser().resolve()

    if p.exists():
        if not p.is_dir():
            raise RuntimeError(f"not a directory: {p}")
        shutil.rmtree(p)

    print(
        json.dumps(
            {
                "ok": True,
                "action": "delete_store",
                "path": str(p),
            },
            ensure_ascii=False,
        )
    )


def delete_collection(path: str, name: str) -> None:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise RuntimeError(f"not a directory: {p}")

    client = chromadb.PersistentClient(path=str(p))
    client.delete_collection(name=name)

    print(
        json.dumps(
            {
                "ok": True,
                "action": "delete_collection",
                "path": str(p),
                "name": name,
            },
            ensure_ascii=False,
        )
    )


def list_collections(path: str) -> None:
    p = Path(path).expanduser().resolve()

    if not p.is_dir():
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "list",
                    "path": str(p),
                    "collections": [],
                },
                ensure_ascii=False,
            )
        )
        return

    client = chromadb.PersistentClient(path=str(p))
    names: list[str] = []
    for c in client.list_collections():
        try:
            names.append(c.name)
        except Exception:
            pass

    print(
        json.dumps(
            {
                "ok": True,
                "action": "list",
                "path": str(p),
                "collections": sorted(names),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["create", "delete_store", "delete_collection", "list"],
    )
    parser.add_argument("--path", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--metadata-json", default="")
    args = parser.parse_args()

    if args.action == "create":
        if not args.name:
            raise SystemExit("--name is required for create")
        create_collection(args.path, args.name, args.metadata_json)

    elif args.action == "delete_store":
        delete_store(args.path)

    elif args.action == "delete_collection":
        if not args.name:
            raise SystemExit("--name is required for delete_collection")
        delete_collection(args.path, args.name)

    elif args.action == "list":
        list_collections(args.path)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        print(
            json.dumps(
                {"ok": False, "error": str(e), "error_type": type(e).__name__},
                ensure_ascii=False,
            ),
            file=sys.stdout,
        )
        sys.exit(1)
