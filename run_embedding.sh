#!/bin/bash

# PDF 임베딩 실행 스크립트

# 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

# 가상 환경 활성화
if [ -d "genv" ]; then
    source genv/bin/activate
    echo "가상 환경 활성화됨"
else
    echo "경고: 가상 환경을 찾을 수 없습니다."
fi

# 필요한 패키지 설치 확인 및 설치
if ! python3 -c "import fitz" 2>/dev/null; then
    echo "필요한 패키지를 설치하는 중..."
    pip install -r requirements.txt
fi

# PDF 임베딩 실행
echo "PDF 임베딩 시작..."
python3 manualemb.py

