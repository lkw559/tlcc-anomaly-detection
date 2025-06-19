#!/usr/bin/env python3
"""
논문용 추가 실험 및 분석
- Ablation Study (TLCC vs No-TLCC 비교)
- Statistical Significance Tests
- Performance Stability Analysis
- Computational Efficiency Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import subprocess
import time
from datetime import datetime

def run_ablation_study():
    """Ablation Study: TLCC 효과 검증"""
    print("🔬 Ablation Study 시작: TLCC vs No-TLCC 비교")
    
    datasets = ['WADI', 'SMAP', 'MSL']
    epochs = [10]  # 중간 값으로 고정
    
    ablation_results = []
    
    for dataset in datasets:
        print(f"\n📊 {dataset} 데이터셋 Ablation Study")
        
        # TLCC 사용 (threshold = 0.8, 최적값)
        cmd_tlcc = [
            'python', 'train_original.py',
            '--comment', f'{dataset}_ablation_with_tlcc',
            '--epoch', '10',
            '--bs', '128',
            '--dataset', dataset,
            '--tlcc_threshold', '0.8',
            '--use_true_tlcc', '1',
            '--tlcc_binary', '1'
        ]
        
        # TLCC 미사용 (threshold = 0.0)
        cmd_no_tlcc = [
            'python', 'train_original.py',
            '--comment', f'{dataset}_ablation_no_tlcc',
            '--epoch', '10',
            '--bs', '128',
            '--dataset', dataset,
            '--tlcc_threshold', '0.0',
            '--use_true_tlcc', '1',
            '--tlcc_binary', '1'
        ]
        
        for condition, cmd in [('with_tlcc', cmd_tlcc), ('no_tlcc', cmd_no_tlcc)]:
            print(f"  🚀 실행: {condition}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode == 0:
                    print(f"    ✅ {condition} 완료")
                else:
                    print(f"    ❌ {condition} 실패: {result.stderr}")
            except Exception as e:
                print(f"    ❌ {condition} 오류: {str(e)}")
    
    print("✅ Ablation Study 완료")

def statistical_significance_test(df):
    """통계적 유의성 검정"""
    print("\n" + "="*60)
    print("📊 통계적 유의성 검정")
    print("="*60)
    
    # TLCC threshold 그룹별 성능 비교
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]
    
    print("\n🔍 TLCC Threshold 그룹 간 F1 Score 차이 검정:")
    
    for i, thresh1 in enumerate(thresholds[:-1]):
        for thresh2 in thresholds[i+1:]:
            group1 = df[df['tlcc_threshold'] == thresh1]['best_f1'].values
            group2 = df[df['tlcc_threshold'] == thresh2]['best_f1'].values
            
            if len(group1) > 0 and len(group2) > 0:
                # Welch's t-test (등분산 가정하지 않음)
                statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                                    (len(group2)-1)*np.var(group2, ddof=1)) / 
                                   (len(group1) + len(group2) - 2))
                cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"  {thresh1:.1f} vs {thresh2:.1f}: t={statistic:.3f}, p={p_value:.6f} {significance}, d={cohens_d:.3f}")

def performance_stability_analysis(df):
    """성능 안정성 분석"""
    print("\n" + "="*60)
    print("📊 성능 안정성 분석")
    print("="*60)
    
    # 데이터셋별 성능 변동성 (CV: Coefficient of Variation)
    print("\n📈 데이터셋별 성능 변동성 (CV = std/mean):")
    
    stability_metrics = []
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        f1_mean = dataset_data['best_f1'].mean()
        f1_std = dataset_data['best_f1'].std()
        f1_cv = f1_std / f1_mean if f1_mean > 0 else 0
        
        roc_mean = dataset_data['epsilon_roc_auc'].mean()
        roc_std = dataset_data['epsilon_roc_auc'].std()
        roc_cv = roc_std / roc_mean if roc_mean > 0 else 0
        
        print(f"  {dataset}:")
        print(f"    F1 Score: {f1_mean:.4f} ± {f1_std:.4f} (CV: {f1_cv:.4f})")
        print(f"    ROC-AUC:  {roc_mean:.4f} ± {roc_std:.4f} (CV: {roc_cv:.4f})")
        
        stability_metrics.append({
            'dataset': dataset,
            'f1_cv': f1_cv,
            'roc_cv': roc_cv,
            'experiments': len(dataset_data)
        })
    
    # 가장 안정적인 데이터셋
    stability_df = pd.DataFrame(stability_metrics)
    most_stable = stability_df.loc[stability_df['f1_cv'].idxmin()]
    print(f"\n🏆 가장 안정적인 데이터셋: {most_stable['dataset']} (F1 CV: {most_stable['f1_cv']:.4f})")

def computational_efficiency_analysis(df):
    """계산 효율성 분석"""
    print("\n" + "="*60)
    print("⚡ 계산 효율성 분석")
    print("="*60)
    
    # 에폭별 시간 분석
    print("\n⏱️ 에폭별 평균 훈련 시간:")
    for epoch in sorted(df['epoch'].unique()):
        epoch_data = df[df['epoch'] == epoch]
        avg_time = epoch_data['total_time'].mean()
        std_time = epoch_data['total_time'].std()
        
        print(f"  {epoch:2d} epochs: {avg_time:6.1f} ± {std_time:5.1f} seconds")
    
    # 성능 대비 시간 효율성
    print("\n📊 성능 대비 시간 효율성 (F1/Time):")
    df['efficiency'] = df['best_f1'] / df['total_time']
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        avg_efficiency = dataset_data['efficiency'].mean()
        max_efficiency = dataset_data['efficiency'].max()
        
        print(f"  {dataset}: 평균 {avg_efficiency:.6f}, 최대 {max_efficiency:.6f}")

def create_publication_plots(df, output_dir='publication_plots'):
    """논문용 고품질 플롯 생성"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 스타일 설정
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 1. TLCC Threshold 효과 (Main Result)
    plt.figure(figsize=(10, 6))
    
    threshold_performance = df.groupby('tlcc_threshold').agg({
        'best_f1': ['mean', 'std'],
        'epsilon_roc_auc': ['mean', 'std'],
        'epsilon_pr_auc': ['mean', 'std']
    })
    
    x = threshold_performance.index
    
    plt.errorbar(x, threshold_performance[('best_f1', 'mean')], 
                yerr=threshold_performance[('best_f1', 'std')],
                label='F1 Score', marker='o', linewidth=2, markersize=8)
    
    plt.errorbar(x, threshold_performance[('epsilon_roc_auc', 'mean')], 
                yerr=threshold_performance[('epsilon_roc_auc', 'std')],
                label='ROC-AUC', marker='s', linewidth=2, markersize=8)
    
    plt.errorbar(x, threshold_performance[('epsilon_pr_auc', 'mean')], 
                yerr=threshold_performance[('epsilon_pr_auc', 'std')],
                label='PR-AUC', marker='^', linewidth=2, markersize=8)
    
    plt.xlabel('TLCC Threshold', fontsize=14)
    plt.ylabel('Performance Score', fontsize=14)
    plt.title('Impact of TLCC Threshold on Anomaly Detection Performance', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tlcc_threshold_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 데이터셋별 성능 비교 (Box Plot)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='dataset', y='best_f1')
    plt.title('F1 Score by Dataset')
    plt.ylabel('F1 Score')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='dataset', y='epsilon_roc_auc')
    plt.title('ROC-AUC by Dataset')
    plt.ylabel('ROC-AUC')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='dataset', y='epsilon_pr_auc')
    plt.title('PR-AUC by Dataset')
    plt.ylabel('PR-AUC')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='dataset', y='epsilon_mcc')
    plt.title('MCC by Dataset')
    plt.ylabel('MCC')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 논문용 플롯 저장 완료: {output_dir}/")

def generate_latex_table(df, output_file='results_table.tex'):
    """LaTeX 테이블 생성"""
    
    # 데이터셋별 최고 성능 요약
    best_results = []
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        best_row = dataset_data.loc[dataset_data['best_f1'].idxmax()]
        
        best_results.append({
            'Dataset': dataset,
            'TLCC Threshold': best_row['tlcc_threshold'],
            'Epochs': int(best_row['epoch']),
            'F1 Score': f"{best_row['best_f1']:.4f}",
            'ROC-AUC': f"{best_row['epsilon_roc_auc']:.4f}",
            'PR-AUC': f"{best_row['epsilon_pr_auc']:.4f}",
            'MCC': f"{best_row['epsilon_mcc']:.4f}"
        })
    
    best_df = pd.DataFrame(best_results)
    
    # LaTeX 테이블 생성
    latex_table = best_df.to_latex(index=False, column_format='lcccccc')
    
    with open(output_file, 'w') as f:
        f.write("% 최고 성능 결과 테이블\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Best Performance Results by Dataset}\n")
        f.write("\\label{tab:best_results}\n")
        f.write(latex_table)
        f.write("\\end{table}\n")
    
    print(f"📄 LaTeX 테이블 생성: {output_file}")

def main():
    """논문용 추가 분석 실행"""
    print("📚 논문용 추가 실험 및 분석 시작")
    
    # 1. Ablation Study 실행 (옵션)
    # run_ablation_study()
    
    # 2. 기존 결과 분석
    import glob
    result_files = glob.glob("comprehensive_experiment_results_*.csv")
    if result_files:
        latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # 통계 분석
        statistical_significance_test(df)
        performance_stability_analysis(df)
        computational_efficiency_analysis(df)
        
        # 논문용 시각화
        create_publication_plots(df)
        
        # LaTeX 테이블 생성
        generate_latex_table(df)
        
        print("\n🎉 논문용 분석 완료!")
    else:
        print("❌ 실험 결과 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
