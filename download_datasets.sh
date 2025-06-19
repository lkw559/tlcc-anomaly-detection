#!/bin/bash
"""
데이터셋 다운로드 및 설정 스크립트
WADI, SMAP, MSL, SMD 데이터셋을 다운로드하고 전처리합니다.
"""

echo "TLCC 이상 탐지 데이터셋 다운로드 시작..."

# 데이터셋 디렉토리 생성
mkdir -p datasets

# SMAP, MSL 데이터셋 다운로드 (NASA Telemetry)
echo "SMAP, MSL 데이터셋 다운로드 중..."
cd datasets
if [ ! -f "data.zip" ]; then
    wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
    unzip data.zip
    rm data.zip
fi

# 레이블 파일 다운로드
cd data
if [ ! -f "labeled_anomalies.csv" ]; then
    wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
fi

# 불필요한 폴더 정리
rm -rf 2018-05-19_15.00.10 2>/dev/null || true

cd ../..

echo "데이터셋 다운로드 완료!"
echo ""
echo "다음 단계:"
echo "1. WADI 데이터셋은 별도로 다운로드해야 합니다:"
echo "   https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/"
echo ""
echo "2. 데이터 전처리 실행:"
echo "   python src/data/preprocess.py --dataset SMAP"
echo "   python src/data/preprocess.py --dataset MSL" 
echo "   python src/data/preprocess.py --dataset SMD"
echo "   python src/data/preprocess.py --dataset WADI"
echo ""
echo "3. 실험 시작:"
echo "   python main.py train --dataset WADI --epochs 5"
