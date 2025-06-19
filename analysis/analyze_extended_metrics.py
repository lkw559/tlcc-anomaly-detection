#!/usr/bin/env python3
"""
기존 실험 결과에 추가 평가지표 (ROC-AUC, PR-AUC, MCC) 적용하는 스크립트
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eval_methods import calculate_additional_metrics, print_extended_metrics

def load_experiment_data(experiment_dir):
    """실험 결과 데이터 로드"""
    test_output_path = os.path.join(experiment_dir, 'test_output.pkl')
    config_path = os.path.join(experiment_dir, 'config.txt')
    summary_path = os.path.join(experiment_dir, 'summary.txt')
    
    if not all(os.path.exists(p) for p in [test_output_path, config_path, summary_path]):
        return None
    
    try:
        with open(test_output_path, 'rb') as f:
            test_output = pickle.load(f)
        with open(config_path, 'r') as f:
            config = json.load(f)
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        return test_output, config, summary
    except Exception as e:
        print(f"Error loading {experiment_dir}: {e}")
        return None

def calculate_extended_metrics_for_experiment(experiment_dir):
    """단일 실험에 대해 확장된 지표 계산"""
    
    data = load_experiment_data(experiment_dir)
    if data is None:
        return None
    
    test_output, config, summary = data
    
    # 필요한 데이터 추출
    if 'A_Score_Global' not in test_output or 'A_True_Global' not in test_output:
        print(f"Warning: Missing required data in {experiment_dir}")
        return None
    
    scores = test_output['A_Score_Global']
    labels = test_output['A_True_Global']
    
    # 기존 임계값 사용
    epsilon_threshold = summary['epsilon_result']['threshold']
    
    # 예측값 계산
    from eval_methods import adjust_predicts
    pred_result = adjust_predicts(scores, labels, epsilon_threshold, calc_latency=False)
    if isinstance(pred_result, tuple):
        pred = pred_result[0]
    else:
        pred = pred_result
    
    # 추가 지표 계산
    additional_metrics = calculate_additional_metrics(labels, pred, scores)
    
    # 결과 통합
    result = {
        'experiment': os.path.basename(experiment_dir),
        'comment': config.get('comment', 'N/A'),
        'tlcc_threshold': config.get('tlcc_threshold', 1.0),
        'epochs': config.get('epochs', 1),
        'tlcc_binary': config.get('tlcc_binary', False),
        
        # 기존 지표
        'f1': summary['epsilon_result']['f1'],
        'precision': summary['epsilon_result']['precision'],
        'recall': summary['epsilon_result']['recall'],
        'TP': summary['epsilon_result']['TP'],
        'FP': summary['epsilon_result']['FP'],
        'TN': summary['epsilon_result']['TN'],
        'FN': summary['epsilon_result']['FN'],
        'threshold': epsilon_threshold,
        
        # 추가 지표
        **additional_metrics
    }
    
    return result

def analyze_all_experiments():
    """모든 WADI 실험에 대해 확장된 지표 분석"""
    
    base_path = '/home/timeseries/[filtered_data]S53CCF/Smartfactory/experiment_WaDi/output/WADI/'
    experiment_dirs = [d for d in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, d)) and d != 'logs']
    experiment_dirs.sort(reverse=True)  # 최신순
    
    print("🔍 모든 WADI 실험에 확장된 평가지표 적용 중...")
    print(f"총 {len(experiment_dirs)}개의 실험 발견")
    
    results = []
    
    for i, exp_dir in enumerate(experiment_dirs):
        full_path = os.path.join(base_path, exp_dir)
        print(f"\n{i+1}/{len(experiment_dirs)} 처리 중: {exp_dir}")
        
        result = calculate_extended_metrics_for_experiment(full_path)
        if result:
            results.append(result)
            
            # 주요 지표 출력
            print(f"   F1: {result['f1']:.4f}, ROC-AUC: {result['roc_auc']:.4f}, "
                  f"PR-AUC: {result['pr_auc']:.4f}, MCC: {result['mcc']:.4f}")
        else:
            print(f"   실패: 데이터 로드 오류")
    
    if not results:
        print("분석할 실험 결과가 없습니다.")
        return None
    
    return pd.DataFrame(results)

def create_extended_metrics_comparison(df):
    """확장된 지표로 실험 비교 시각화"""
    
    if len(df) < 2:
        print("시각화를 위한 충분한 데이터가 없습니다.")
        return
    
    # 한글 폰트 설정 (옵션)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('확장된 평가지표로 본 WADI 실험 결과 비교', fontsize=16)
    
    # 1. TLCC threshold별 F1 vs ROC-AUC
    ax1 = axes[0, 0]
    for threshold in sorted(df['tlcc_threshold'].unique()):
        subset = df[df['tlcc_threshold'] == threshold]
        ax1.scatter(subset['f1'], subset['roc_auc'], 
                   label=f'TLCC {threshold}', alpha=0.7, s=100)
    ax1.set_xlabel('F1 Score')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('F1 vs ROC-AUC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. TLCC threshold별 F1 vs PR-AUC
    ax2 = axes[0, 1]
    for threshold in sorted(df['tlcc_threshold'].unique()):
        subset = df[df['tlcc_threshold'] == threshold]
        ax2.scatter(subset['f1'], subset['pr_auc'], 
                   label=f'TLCC {threshold}', alpha=0.7, s=100)
    ax2.set_xlabel('F1 Score')
    ax2.set_ylabel('PR-AUC')
    ax2.set_title('F1 vs PR-AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. TLCC threshold별 F1 vs MCC
    ax3 = axes[0, 2]
    for threshold in sorted(df['tlcc_threshold'].unique()):
        subset = df[df['tlcc_threshold'] == threshold]
        ax3.scatter(subset['f1'], subset['mcc'], 
                   label=f'TLCC {threshold}', alpha=0.7, s=100)
    ax3.set_xlabel('F1 Score')
    ax3.set_ylabel('MCC')
    ax3.set_title('F1 vs MCC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 지표별 박스플롯
    metrics_to_plot = ['f1', 'roc_auc', 'pr_auc', 'mcc']
    metric_names = ['F1 Score', 'ROC-AUC', 'PR-AUC', 'MCC']
    
    for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        if i < 3:  # 첫 번째 행의 나머지 공간 사용하지 않음
            continue
        ax = axes[1, i-3] if i >= 3 else axes[1, i]
        
        # TLCC threshold별 박스플롯
        data_for_box = []
        labels_for_box = []
        
        for threshold in sorted(df['tlcc_threshold'].unique()):
            subset = df[df['tlcc_threshold'] == threshold]
            if len(subset) > 0:
                data_for_box.append(subset[metric].values)
                labels_for_box.append(f'TLCC {threshold}')
        
        if data_for_box:
            ax.boxplot(data_for_box, labels=labels_for_box)
            ax.set_title(f'{name} by TLCC Threshold')
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
    
    # 5. 상관관계 히트맵
    ax5 = axes[1, 0]
    correlation_metrics = ['f1', 'precision', 'recall', 'roc_auc', 'pr_auc', 'mcc']
    corr_data = df[correlation_metrics].corr()
    
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax5)
    ax5.set_title('지표 간 상관관계')
    
    # 6. 종합 성능 레이더 차트 (상위 3개 실험)
    ax6 = axes[1, 1]
    
    # 상위 3개 실험 선택 (F1 기준)
    top3 = df.nlargest(3, 'f1')
    
    # 레이더 차트 대신 바 차트로 대체
    x_pos = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, (idx, row) in enumerate(top3.iterrows()):
        values = [row[metric] for metric in metrics_to_plot]
        ax6.bar(x_pos + i*width, values, width, 
               label=f"{row['comment'][:20]}...", alpha=0.7)
    
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Score')
    ax6.set_title('Top 3 Experiments Comparison')
    ax6.set_xticks(x_pos + width)
    ax6.set_xticklabels(metric_names, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. TLCC threshold 효과 요약
    ax7 = axes[1, 2]
    
    tlcc_summary = df.groupby('tlcc_threshold')[metrics_to_plot].mean()
    
    x_pos = np.arange(len(tlcc_summary.index))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        ax7.bar(x_pos + i*width, tlcc_summary[metric], width, 
               label=metric_names[i], alpha=0.7)
    
    ax7.set_xlabel('TLCC Threshold')
    ax7.set_ylabel('Average Score')
    ax7.set_title('Average Performance by TLCC Threshold')
    ax7.set_xticks(x_pos + 1.5*width)
    ax7.set_xticklabels([f'TLCC {t}' for t in tlcc_summary.index])
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 파일 저장
    output_path = '/home/timeseries/[filtered_data]S53CCF/Smartfactory/experiment_WaDi/extended_metrics_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 시각화 결과 저장: {output_path}")
    
    plt.show()

def print_ranking_by_metrics(df):
    """각 지표별 실험 순위 출력"""
    
    metrics = ['f1', 'roc_auc', 'pr_auc', 'mcc']
    metric_names = ['F1 Score', 'ROC-AUC', 'PR-AUC', 'MCC']
    
    print(f"\n🏆 지표별 실험 순위 (상위 3개)")
    print("=" * 80)
    
    for metric, name in zip(metrics, metric_names):
        print(f"\n📊 {name} 순위:")
        top3 = df.nlargest(3, metric)
        
        for i, (idx, row) in enumerate(top3.iterrows()):
            print(f"   {i+1}위. {row['comment'][:40]}...")
            print(f"        {name}: {row[metric]:.4f}")
            print(f"        TLCC: {row['tlcc_threshold']}, Epochs: {row['epochs']}")
            print(f"        F1: {row['f1']:.4f}, Precision: {row['precision']:.4f}, Recall: {row['recall']:.4f}")

def main():
    """메인 분석 함수"""
    
    print("🔬 기존 실험 결과에 확장된 평가지표 (ROC-AUC, PR-AUC, MCC) 적용")
    print("=" * 80)
    
    # 모든 실험 분석
    df = analyze_all_experiments()
    
    if df is None or len(df) == 0:
        print("분석할 실험 결과가 없습니다.")
        return
    
    print(f"\n✅ 총 {len(df)}개 실험 분석 완료")
    
    # 요약 통계
    print(f"\n📈 지표별 요약 통계:")
    key_metrics = ['f1', 'roc_auc', 'pr_auc', 'mcc']
    summary_stats = df[key_metrics].describe()
    print(summary_stats.round(4))
    
    # TLCC threshold별 평균 성능
    print(f"\n🔗 TLCC Threshold별 평균 성능:")
    tlcc_avg = df.groupby('tlcc_threshold')[key_metrics].mean()
    print(tlcc_avg.round(4))
    
    # 지표별 순위
    print_ranking_by_metrics(df)
    
    # 최고 종합 성능 실험
    print(f"\n🥇 종합 최고 성능 실험:")
    # 모든 지표의 평균으로 종합 점수 계산
    df['composite_score'] = df[key_metrics].mean(axis=1)
    best_overall = df.loc[df['composite_score'].idxmax()]
    
    print(f"   실험: {best_overall['comment']}")
    print(f"   TLCC: {best_overall['tlcc_threshold']}, Epochs: {best_overall['epochs']}")
    print(f"   종합점수: {best_overall['composite_score']:.4f}")
    print(f"   F1: {best_overall['f1']:.4f}, ROC-AUC: {best_overall['roc_auc']:.4f}")
    print(f"   PR-AUC: {best_overall['pr_auc']:.4f}, MCC: {best_overall['mcc']:.4f}")
    
    # 시각화
    create_extended_metrics_comparison(df)
    
    # 결과 저장
    output_csv = '/home/timeseries/[filtered_data]S53CCF/Smartfactory/experiment_WaDi/extended_metrics_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n💾 상세 결과 저장: {output_csv}")
    
    print(f"\n🎉 확장된 평가지표 분석 완료!")

if __name__ == "__main__":
    main()
