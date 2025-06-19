# TLCC-Based Anomaly Detection

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

시계열 데이터에서 TLCC(Time Lagged Cross Correlation) 기반 이상 탐지를 수행하는 딥러닝 프레임워크입니다. MTAD-GAT 아키텍처를 기반으로 하여 다변량 시계열 데이터의 이상을 효과적으로 탐지합니다.

## 🚀 주요 기능

- **확장된 평가 지표**: ROC-AUC, PR-AUC, MCC 포함한 종합적인 성능 평가
- **TLCC 기반 상관관계 분석**: 시간 지연 교차 상관관계를 활용한 특성 선택
- **다중 데이터셋 지원**: WADI, SMAP, MSL, SMD 데이터셋 지원
- **포괄적 실험 프레임워크**: 자동화된 실험 실행 및 결과 분석
- **시각화 및 분석 도구**: 실험 결과 시각화 및 통계 분석

## 📊 지원 데이터셋

| 데이터셋 | 설명 | 특성 수 | 이상 비율 |
|---------|------|---------|-----------|
| WADI | 수처리 시설 데이터 | 123 | ~5% |
| SMAP | NASA 위성 데이터 | 25 | ~13% |
| MSL | NASA 화성 탐사선 데이터 | 55 | ~10% |
| SMD | 서버 기계 데이터 | 38 | ~4% |

## 🛠️ 설치 및 설정

### 1. 저장소 클론
```bash
git clone https://github.com/lkw559/tlcc-anomaly-detection.git
cd tlcc-anomaly-detection
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows의 경우: venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 🔧 사용법

### 기본 사용법

#### 단일 모델 훈련
```bash
python main.py train --dataset WADI --epochs 5 --tlcc_threshold 0.5
```

#### 포괄적 실험 실행
```bash
python main.py experiment --datasets WADI SMAP MSL SMD
```

#### 결과 분석
```bash
python main.py analyze --results_dir output/ --output_dir analysis_output/
```

### 고급 사용법

#### 직접 스크립트 실행
```bash
# 단일 실험
python experiments/train_original.py --dataset WADI --epochs 5

# 포괄적 실험
python experiments/comprehensive_experiment_clean.py

# 결과 분석
python experiments/analyze_comprehensive_results.py
```

## 📈 평가 지표

### 기본 지표
- **F1-Score**: 정밀도와 재현율의 조화 평균
- **Precision**: 예측된 이상 중 실제 이상 비율
- **Recall**: 실제 이상 중 탐지된 이상 비율

### 확장 지표
- **ROC-AUC**: Receiver Operating Characteristic Area Under Curve
- **PR-AUC**: Precision-Recall Area Under Curve  
- **MCC**: Matthews Correlation Coefficient

### 성능 등급 기준
| 지표 | 우수 | 양호 | 보통 | 개선필요 |
|------|------|------|------|----------|
| F1-Score | ≥0.8 | ≥0.6 | ≥0.4 | <0.4 |
| ROC-AUC | ≥0.9 | ≥0.8 | ≥0.7 | <0.7 |
| PR-AUC | ≥0.8 | ≥0.6 | ≥0.4 | <0.4 |
| MCC | ≥0.6 | ≥0.4 | ≥0.2 | <0.2 |

## 📁 프로젝트 구조

```
tlcc-anomaly-detection/
├── main.py                    # 메인 진입점
├── requirements.txt           # 의존성 목록
├── setup.py                  # 패키지 설정
├── README.md                 # 프로젝트 문서
├── src/                      # 소스 코드
│   ├── data/                 # 데이터 처리
│   ├── models/               # 모델 정의
│   ├── evaluation/           # 평가 메서드
│   └── utils/               # 유틸리티 함수
├── experiments/             # 실험 스크립트
│   ├── train_original.py    # 단일 모델 훈련
│   ├── comprehensive_experiment_clean.py  # 포괄적 실험
│   └── analyze_comprehensive_results.py   # 결과 분석
├── analysis/               # 분석 도구
├── notebooks/             # Jupyter 노트북
├── config/               # 설정 파일
└── docs/                # 문서
```

## 🧪 실험 결과 예시

### MSL 데이터셋 결과 (최고 성능)
```
=== Extended Metrics Results ===
Dataset: MSL, Epochs: 3, TLCC Threshold: 0.0

Epsilon Method:
- F1: 0.954 (우수)
- Precision: 0.923, Recall: 0.987
- ROC-AUC: 0.982 (우수)
- PR-AUC: 0.945 (우수)
- MCC: 0.913 (우수)
```

### WADI 데이터셋 결과
```
=== Extended Metrics Results ===
Dataset: WADI, Epochs: 1, TLCC Threshold: 0.5

Epsilon Method:
- F1: 0.730 (양호)
- Precision: 0.651, Recall: 0.825
- ROC-AUC: 0.912 (우수)
- PR-AUC: 0.687 (양호)
- MCC: 0.623 (우수)
```

## 🧬 모델 아키텍처

이 프로젝트는 MTAD-GAT를 기반으로 한 다음 구조를 사용합니다:

1. **1D Convolution Layer**: 시간 차원에서 데이터 스무딩
2. **Dual GAT Layers**: 
   - Feature-oriented GAT: 특성 간 의존성 캡처
   - Time-oriented GAT: 시간 단계 간 의존성 캡처
3. **GRU Layer**: 장기 순차 패턴 학습
4. **Dual Output**: 예측 및 재구성 모델

### TLCC 기반 특성 선택

시간 지연 교차 상관관계(TLCC)를 사용하여 중요한 특성을 선택합니다.

## 📊 성능 벤치마크

### 최적 실험 결과 (True TLCC 사용)

| 데이터셋 | 최적 F1 | 최적 TLCC | 에포크 | ROC-AUC | PR-AUC | MCC |
|---------|---------|-----------|--------|---------|--------|-----|
| MSL | **0.954** | 0.0 | 3 | 0.982 | 0.945 | 0.913 |
| SMAP | **0.801** | 0.3 | 1 | 0.934 | 0.789 | 0.756 |
| SMD | **0.774** | 0.0 | 3 | 0.901 | 0.723 | 0.698 |
| WADI | **0.730** | 0.5 | 1 | 0.912 | 0.687 | 0.623 |

### 주요 발견사항
- **MSL**: TLCC 0.0에서 최고 성능 (F1=0.954), 장기 학습 필요
- **SMAP**: TLCC 0.3에서 최적, 단기 학습으로 충분
- **SMD**: POT 방법에서 뛰어난 성능, TLCC 0.0 최적
- **WADI**: 중간 정도 시간 지연(0.5) 필요

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 브랜치를 만듭니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📖 참고 문헌

- Zhao, H., Wang, Y., Duan, J., Huang, C., Cao, D., Tong, Y., ... & Zhu, J. (2020). Multivariate time-series anomaly detection via graph attention network. *IEEE International Conference on Data Mining (ICDM)*.
- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018). Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. *ACM SIGKDD*.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 영감을 받았습니다:
- [MTAD-GAT](https://github.com/ML4ITS/mtad-gat-pytorch)
- [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
- [TelemAnom](https://github.com/khundman/telemanom)
