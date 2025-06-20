#!/usr/bin/env python3
"""
진짜 TLCC (Time-Lagged Cross-Correlation) 구현
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import correlate
import warnings

def true_time_lagged_cross_correlation(x, y, max_lag=10):
    """
    진짜 Time-Lagged Cross-Correlation 계산
    
    Args:
        x, y: 시계열 데이터
        max_lag: 최대 lag 범위
        
    Returns:
        최대 cross-correlation 값과 해당 lag
    """
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    # NaN 제거
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 10:
        return 0.0, 0
    
    for lag in lags:
        if lag == 0:
            # lag 0: 일반적인 correlation
            corr, _ = pearsonr(x_clean, y_clean)
        elif lag > 0:
            # 양의 lag: x가 y를 lag만큼 뒤처짐 (y[t] vs x[t-lag])
            if len(x_clean) > lag:
                corr, _ = pearsonr(x_clean[:-lag], y_clean[lag:])
            else:
                corr = 0.0
        else:  # lag < 0
            # 음의 lag: x가 y를 |lag|만큼 앞섬 (y[t] vs x[t+|lag|])
            abs_lag = abs(lag)
            if len(y_clean) > abs_lag:
                corr, _ = pearsonr(x_clean[abs_lag:], y_clean[:-abs_lag])
            else:
                corr = 0.0
        
        if not np.isnan(corr):
            correlations.append(abs(corr))  # 절댓값으로 최대값 찾기
        else:
            correlations.append(0.0)
    
    # 최대 correlation과 해당 lag 반환
    max_corr_idx = np.argmax(correlations)
    max_corr = correlations[max_corr_idx]
    best_lag = lags[max_corr_idx]
    
    return max_corr, best_lag

def cross_correlate_tlcc(data_df, target_series, lag_idx=1, max_lag=10):
    """
    진짜 TLCC 기반 cross-correlation 계산
    
    Args:
        data_df: 전체 데이터프레임
        target_series: 타겟 시계열
        lag_idx: lag index (사용되지 않음, 호환성 위해 유지)
        max_lag: 최대 lag 범위
        
    Returns:
        각 컬럼과 target_series 간의 최대 TLCC 값들
    """
    correlations = {}
    target_values = target_series.values
    
    for col in data_df.columns:
        if col == target_series.name:
            correlations[col] = 1.0
        else:
            try:
                series_values = data_df[col].values
                max_corr, best_lag = true_time_lagged_cross_correlation(
                    target_values, series_values, max_lag=max_lag
                )
                correlations[col] = max_corr
            except Exception as e:
                print(f"Warning: TLCC calculation failed for {col}: {e}")
                correlations[col] = 0.0
    
    return pd.Series(correlations)

def columns_by_max_cross_correlation_tlcc(data_df, max_lag=10, dataset_name=None, output_dir=None, **kwargs):
    """
    진짜 TLCC 기반 상관관계 행렬 계산 (캐싱 기능 포함)
    
    Args:
        data_df: Input DataFrame
        max_lag: 최대 time lag
        dataset_name: 데이터셋 이름 (캐싱용)
        output_dir: 출력 디렉토리 (TLCC 결과 저장용)
        
    Returns:
        TLCC 기반 correlation matrix
    """
    import os
    
    # 캐시 파일 경로 설정
    if output_dir and dataset_name:
        os.makedirs(output_dir, exist_ok=True)
        cache_file = os.path.join(output_dir, f"tlcc_correlation_matrix_{dataset_name}_lag{max_lag}.csv")
        lag_cache_file = os.path.join(output_dir, f"tlcc_lag_matrix_{dataset_name}_lag{max_lag}.csv")
    else:
        # 기본 경로 (이전 동작 유지)
        cache_file = f"tlcc_correlation_matrix_lag{max_lag}.csv"
        lag_cache_file = f"tlcc_lag_matrix_lag{max_lag}.csv"
    
    # 기존 캐시 파일이 있는지 확인
    if os.path.exists(cache_file) and os.path.exists(lag_cache_file):
        print(f"📁 Loading cached TLCC results from {cache_file}")
        try:
            corr_df = pd.read_csv(cache_file, index_col=0)
            lag_df = pd.read_csv(lag_cache_file, index_col=0)
            
            # 데이터 형태 검증 (shape만 확인, 컬럼명은 유연하게 처리)
            if corr_df.shape == (data_df.shape[1], data_df.shape[1]):
                # 컬럼명이 다르면 현재 데이터에 맞게 조정
                if list(corr_df.columns) != list(data_df.columns):
                    print(f"📝 Adjusting cached column names to match current data...")
                    corr_df.columns = data_df.columns
                    corr_df.index = data_df.columns
                    lag_df.columns = data_df.columns  
                    lag_df.index = data_df.columns
                
                print(f"✅ Successfully loaded cached TLCC matrix. Shape: {corr_df.shape}")
                print(f"TLCC correlation range: {corr_df.values.min():.4f} to {corr_df.values.max():.4f}")
                print(f"Lag range: {lag_df.values.min():.0f} to {lag_df.values.max():.0f}")
                return corr_df
            else:
                print("⚠️ Cache file structure doesn't match current data, recalculating...")
        except Exception as e:
            print(f"⚠️ Error loading cache file: {e}, recalculating...")
    
    print(f"🔄 Computing TRUE Time-Lagged Cross-Correlations (max_lag={max_lag})...")
    if dataset_name:
        print(f"Dataset: {dataset_name}")
    
    n_features = data_df.shape[1]
    corr_matrix = np.eye(n_features)  # 대각선은 1.0
    lag_matrix = np.zeros((n_features, n_features))  # 최적 lag 저장
    
    # 모든 feature pair에 대해 TLCC 계산
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                try:
                    series_i = data_df.iloc[:, i].values
                    series_j = data_df.iloc[:, j].values
                    
                    max_corr, best_lag = true_time_lagged_cross_correlation(
                        series_i, series_j, max_lag=max_lag
                    )
                    
                    corr_matrix[i, j] = max_corr
                    lag_matrix[i, j] = best_lag
                    
                except Exception as e:
                    print(f"Warning: TLCC error between {i} and {j}: {e}")
                    corr_matrix[i, j] = 0.0
                    lag_matrix[i, j] = 0
        
        # 진행상황 출력
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_features} features...")
     # DataFrame으로 변환
    corr_df = pd.DataFrame(corr_matrix, 
                          index=data_df.columns, 
                          columns=data_df.columns)

    lag_df = pd.DataFrame(lag_matrix,
                         index=data_df.columns,
                         columns=data_df.columns)

    print(f"TLCC matrix computed. Shape: {corr_df.shape}")
    print(f"TLCC correlation range: {corr_df.values.min():.4f} to {corr_df.values.max():.4f}")
    print(f"Lag range: {lag_df.values.min():.0f} to {lag_df.values.max():.0f}")

    # 결과 저장 (캐싱)
    try:
        corr_df.to_csv(cache_file)
        lag_df.to_csv(lag_cache_file)
        print(f"💾 TLCC results saved to:")
        print(f"  Correlation matrix: {cache_file}")
        print(f"  Optimal lag matrix: {lag_cache_file}")
    except Exception as e:
        print(f"⚠️ Error saving TLCC cache: {e}")

    return corr_df

def test_tlcc_implementation():
    """TLCC 구현 테스트"""
    print("=" * 60)
    print("🧪 TLCC (Time-Lagged Cross-Correlation) 구현 테스트")
    print("=" * 60)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    t = np.arange(n_samples)
    
    # 시계열 1: 기본 신호
    x1 = np.sin(0.1 * t) + 0.1 * np.random.randn(n_samples)
    
    # 시계열 2: x1을 5 time step 지연시킨 신호 + 노이즈
    lag_true = 5
    x2 = np.zeros_like(x1)
    x2[lag_true:] = x1[:-lag_true] + 0.2 * np.random.randn(n_samples - lag_true)
    
    # 시계열 3: 무관한 신호
    x3 = np.random.randn(n_samples)
    
    print("테스트 시나리오:")
    print(f"  x1: 기본 신호")
    print(f"  x2: x1을 {lag_true} step 지연시킨 신호")
    print(f"  x3: 무관한 노이즈 신호")
    
    # TLCC 계산
    print(f"\n📊 TLCC 계산 결과:")
    
    # x1 vs x2 (지연된 관계가 있어야 함)
    max_corr_12, best_lag_12 = true_time_lagged_cross_correlation(x1, x2, max_lag=10)
    print(f"  x1 vs x2: 최대 correlation = {max_corr_12:.4f}, 최적 lag = {best_lag_12}")
    
    # x1 vs x3 (관계가 없어야 함)
    max_corr_13, best_lag_13 = true_time_lagged_cross_correlation(x1, x3, max_lag=10)
    print(f"  x1 vs x3: 최대 correlation = {max_corr_13:.4f}, 최적 lag = {best_lag_13}")
    
    # x2 vs x3 (관계가 없어야 함)
    max_corr_23, best_lag_23 = true_time_lagged_cross_correlation(x2, x3, max_lag=10)
    print(f"  x2 vs x3: 최대 correlation = {max_corr_23:.4f}, 최적 lag = {best_lag_23}")
    
    # 일반 correlation과 비교
    corr_normal_12 = abs(np.corrcoef(x1, x2)[0, 1])
    corr_normal_13 = abs(np.corrcoef(x1, x3)[0, 1])
    
    print(f"\n📈 일반 correlation과 비교:")
    print(f"  x1 vs x2: TLCC = {max_corr_12:.4f}, 일반 = {corr_normal_12:.4f}")
    print(f"  x1 vs x3: TLCC = {max_corr_13:.4f}, 일반 = {corr_normal_13:.4f}")
    
    # 결과 검증
    print(f"\n✅ 검증 결과:")
    if best_lag_12 == lag_true:
        print(f"  🎯 올바른 lag 감지! (예상: {lag_true}, 감지: {best_lag_12})")
    else:
        print(f"  ⚠️  lag 감지 오차 (예상: {lag_true}, 감지: {best_lag_12})")
    
    if max_corr_12 > max_corr_13:
        print(f"  🎯 관련 있는 신호에서 더 높은 correlation!")
    else:
        print(f"  ⚠️  correlation 구분 실패")
        
    if max_corr_12 > corr_normal_12:
        print(f"  🎯 TLCC가 일반 correlation보다 높은 값 검출!")
    else:
        print(f"  ℹ️  일반 correlation과 유사한 수준")
    
    return max_corr_12, best_lag_12

if __name__ == "__main__":
    test_tlcc_implementation()
