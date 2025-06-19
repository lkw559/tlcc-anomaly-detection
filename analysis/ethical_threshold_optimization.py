#!/usr/bin/env python3
"""
연구윤리에 맞는 임계값 최적화 방법
Cross-validation을 통한 하이퍼파라미터 튜닝
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

def ethical_threshold_optimization(train_scores, train_labels=None, method='cv'):
    """
    연구윤리에 맞는 임계값 최적화
    
    Args:
        train_scores: 훈련 데이터의 anomaly scores
        train_labels: 훈련 데이터의 labels (있다면)
        method: 'cv' (cross-validation) 또는 'heuristic'
    
    Returns:
        최적 percentile 값
    """
    
    if method == 'cv' and train_labels is not None:
        return _cross_validation_optimization(train_scores, train_labels)
    else:
        return _heuristic_optimization(train_scores)

def _cross_validation_optimization(train_scores, train_labels):
    """
    Cross-validation을 통한 윤리적 최적화
    """
    print("📊 Cross-Validation 기반 임계값 최적화 수행")
    
    # Time series split (시계열 데이터 특성 고려)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 테스트할 percentile 후보들 (사전 정의)
    percentile_candidates = [90, 92, 94, 95, 96, 97, 98, 99]
    
    best_percentile = 95  # 기본값
    best_cv_score = 0
    
    print("Percentile | CV F1 Score")
    print("-" * 25)
    
    for percentile in percentile_candidates:
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(train_scores):
            # Train/Validation split
            train_fold = train_scores[train_idx]
            val_scores = train_scores[val_idx]
            val_labels = train_labels[val_idx]
            
            # 임계값 계산 (train_fold만 사용)
            threshold = np.percentile(train_fold, percentile)
            
            # Validation에서 평가
            pred = (val_scores > threshold).astype(int)
            f1 = f1_score(val_labels, pred, zero_division=0)
            cv_scores.append(f1)
        
        mean_cv_score = np.mean(cv_scores)
        print(f"{percentile:8d}   | {mean_cv_score:.4f}")
        
        if mean_cv_score > best_cv_score:
            best_cv_score = mean_cv_score
            best_percentile = percentile
    
    print(f"\n🏆 Cross-validation 최적 percentile: {best_percentile}")
    print(f"🎯 Cross-validation F1 score: {best_cv_score:.4f}")
    
    return best_percentile

def _heuristic_optimization(train_scores):
    """
    휴리스틱 기반 최적화 (라벨이 없을 때)
    """
    print("🔍 Heuristic 기반 임계값 최적화 수행")
    
    # 통계적 특성 분석
    stats = {
        'mean': np.mean(train_scores),
        'std': np.std(train_scores),
        'median': np.median(train_scores),
        'q90': np.percentile(train_scores, 90),
        'q95': np.percentile(train_scores, 95),
        'q99': np.percentile(train_scores, 99),
    }
    
    print(f"Score 통계:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")
    
    # 휴리스틱 규칙들
    rules = {
        'conservative': 99,  # 보수적: 99th percentile
        'balanced': 95,      # 균형적: 95th percentile  
        'sensitive': 90,     # 민감한: 90th percentile
    }
    
    # 데이터 분포에 따른 선택
    coefficient_of_variation = stats['std'] / stats['mean']
    
    if coefficient_of_variation < 0.3:
        # 분산이 작으면 더 민감하게
        recommended = 'sensitive'
    elif coefficient_of_variation > 0.8:
        # 분산이 크면 더 보수적으로
        recommended = 'conservative'
    else:
        # 중간 정도면 균형적으로
        recommended = 'balanced'
    
    selected_percentile = rules[recommended]
    
    print(f"\n📊 분포 특성:")
    print(f"  Coefficient of Variation: {coefficient_of_variation:.3f}")
    print(f"  권장 전략: {recommended}")
    print(f"  선택된 percentile: {selected_percentile}")
    
    return selected_percentile

def validate_threshold_method(train_scores, test_scores, test_labels, percentile):
    """
    선택된 임계값 방법의 성능 검증
    """
    threshold = np.percentile(train_scores, percentile)
    pred = (test_scores > threshold).astype(int)
    
    if test_labels is not None:
        f1 = f1_score(test_labels, pred, zero_division=0)
        print(f"\n✅ 최종 테스트 성능:")
        print(f"  Percentile: {percentile}")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Test F1 Score: {f1:.4f}")
        return f1
    else:
        print(f"\n📋 선택된 임계값:")
        print(f"  Percentile: {percentile}")
        print(f"  Threshold: {threshold:.6f}")
        return threshold

# 사용 예시
if __name__ == "__main__":
    # 시뮬레이션 데이터
    np.random.seed(42)
    n_samples = 10000
    
    # 정상 데이터 (낮은 점수)
    normal_scores = np.random.exponential(0.1, int(n_samples * 0.95))
    normal_labels = np.zeros(len(normal_scores))
    
    # 이상 데이터 (높은 점수)
    anomaly_scores = np.random.exponential(0.3, int(n_samples * 0.05)) + 0.2
    anomaly_labels = np.ones(len(anomaly_scores))
    
    # 데이터 합치기
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([normal_labels, anomaly_labels])
    
    # 시간순으로 섞기 (현실적으로)
    idx = np.random.permutation(len(all_scores))
    all_scores = all_scores[idx]
    all_labels = all_labels[idx]
    
    # Train/Test split (시계열이므로 시간순)
    split_point = int(len(all_scores) * 0.7)
    train_scores = all_scores[:split_point]
    train_labels = all_labels[:split_point]
    test_scores = all_scores[split_point:]
    test_labels = all_labels[split_point:]
    
    print("=" * 60)
    print("연구윤리에 맞는 임계값 최적화 예시")
    print("=" * 60)
    
    # 윤리적 최적화 수행
    optimal_percentile = ethical_threshold_optimization(
        train_scores, train_labels, method='cv'
    )
    
    # 최종 검증
    validate_threshold_method(
        train_scores, test_scores, test_labels, optimal_percentile
    )
