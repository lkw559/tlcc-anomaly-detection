#!/usr/bin/env python3
"""
포괄적 실험 결과 분석 및 시각화 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_results(filename):
    """실험 결과 CSV 파일 로드"""
    try:
        df = pd.read_csv(filename)
        print(f"✅ 실험 결과 로드 완료: {len(df)}개 실험")
        return df
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {filename}")
        return None

def analyze_by_dataset(df):
    """데이터셋별 성능 분석"""
    print("\n" + "="*60)
    print("📊 데이터셋별 성능 분석")
    print("="*60)
    
    metrics = ['best_f1', 'epsilon_roc_auc', 'epsilon_pr_auc', 'epsilon_mcc', 
               'pot_roc_auc', 'pot_pr_auc', 'pot_mcc']
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        print(f"\n🔬 {dataset} 데이터셋 ({len(dataset_data)}개 실험)")
        print("-" * 40)
        
        for metric in metrics:
            if metric in dataset_data.columns:
                mean_val = dataset_data[metric].mean()
                std_val = dataset_data[metric].std()
                max_val = dataset_data[metric].max()
                print(f"   {metric:15}: {mean_val:.4f} ± {std_val:.4f} (max: {max_val:.4f})")

def analyze_by_tlcc_threshold(df):
    """TLCC threshold별 성능 분석"""
    print("\n" + "="*60)
    print("📊 TLCC Threshold별 성능 분석")
    print("="*60)
    
    # TLCC threshold별 평균 성능
    threshold_analysis = df.groupby('tlcc_threshold').agg({
        'best_f1': ['mean', 'std', 'max'],
        'epsilon_roc_auc': ['mean', 'std', 'max'],
        'epsilon_pr_auc': ['mean', 'std', 'max'],
        'epsilon_mcc': ['mean', 'std', 'max']
    }).round(4)
    
    print("\n📈 TLCC Threshold별 평균 성능:")
    print(threshold_analysis)
    
    # 최고 성능 TLCC threshold 찾기
    best_threshold = df.groupby('tlcc_threshold')['best_f1'].mean().idxmax()
    best_performance = df.groupby('tlcc_threshold')['best_f1'].mean().max()
    
    print(f"\n🏆 최고 성능 TLCC Threshold: {best_threshold} (평균 F1: {best_performance:.4f})")

def analyze_by_epoch(df):
    """Epoch별 성능 분석"""
    print("\n" + "="*60)
    print("📊 Epoch별 성능 분석")
    print("="*60)
    
    epoch_analysis = df.groupby('epoch').agg({
        'best_f1': ['mean', 'std', 'max'],
        'epsilon_roc_auc': ['mean', 'std', 'max'],
        'epsilon_pr_auc': ['mean', 'std', 'max'],
        'total_time': ['mean', 'std']
    }).round(4)
    
    print("\n📈 Epoch별 성능 및 시간:")
    print(epoch_analysis)

def create_performance_visualizations(df, output_dir='analysis_plots'):
    """성능 시각화 생성"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. 데이터셋별 성능 비교
    plt.figure(figsize=(15, 12))
    
    # F1 Score 비교
    plt.subplot(2, 3, 1)
    sns.boxplot(data=df, x='dataset', y='best_f1')
    plt.title('F1 Score by Dataset')
    plt.xticks(rotation=45)
    
    # ROC-AUC 비교
    plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x='dataset', y='epsilon_roc_auc')
    plt.title('ROC-AUC by Dataset')
    plt.xticks(rotation=45)
    
    # PR-AUC 비교
    plt.subplot(2, 3, 3)
    sns.boxplot(data=df, x='dataset', y='epsilon_pr_auc')
    plt.title('PR-AUC by Dataset')
    plt.xticks(rotation=45)
    
    # TLCC Threshold별 성능
    plt.subplot(2, 3, 4)
    tlcc_performance = df.groupby('tlcc_threshold')['best_f1'].mean()
    plt.plot(tlcc_performance.index, tlcc_performance.values, 'o-')
    plt.xlabel('TLCC Threshold')
    plt.ylabel('Average F1 Score')
    plt.title('Performance vs TLCC Threshold')
    plt.grid(True)
    
    # Epoch별 성능
    plt.subplot(2, 3, 5)
    epoch_performance = df.groupby('epoch')['best_f1'].mean()
    plt.plot(epoch_performance.index, epoch_performance.values, 's-')
    plt.xlabel('Epochs')
    plt.ylabel('Average F1 Score')
    plt.title('Performance vs Epochs')
    plt.grid(True)
    
    # 실행 시간 분석
    plt.subplot(2, 3, 6)
    sns.boxplot(data=df, x='epoch', y='total_time')
    plt.title('Training Time by Epochs')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_heatmap_analysis(df, output_dir='analysis_plots'):
    """히트맵 분석 생성"""
    
    # 데이터셋 x TLCC threshold 히트맵
    plt.figure(figsize=(12, 8))
    
    # F1 Score 히트맵
    plt.subplot(2, 2, 1)
    heatmap_data = df.groupby(['dataset', 'tlcc_threshold'])['best_f1'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('F1 Score: Dataset vs TLCC Threshold')
    
    # ROC-AUC 히트맵
    plt.subplot(2, 2, 2)
    heatmap_data = df.groupby(['dataset', 'tlcc_threshold'])['epsilon_roc_auc'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('ROC-AUC: Dataset vs TLCC Threshold')
    
    # PR-AUC 히트맵
    plt.subplot(2, 2, 3)
    heatmap_data = df.groupby(['dataset', 'tlcc_threshold'])['epsilon_pr_auc'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('PR-AUC: Dataset vs TLCC Threshold')
    
    # MCC 히트맵
    plt.subplot(2, 2, 4)
    heatmap_data = df.groupby(['dataset', 'tlcc_threshold'])['epsilon_mcc'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('MCC: Dataset vs TLCC Threshold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def find_best_configurations(df):
    """최고 성능 설정 찾기"""
    print("\n" + "="*60)
    print("🏆 최고 성능 설정 분석")
    print("="*60)
    
    # 전체 최고 성능
    best_overall = df.loc[df['best_f1'].idxmax()]
    print(f"\n🥇 전체 최고 성능:")
    print(f"   Dataset: {best_overall['dataset']}")
    if best_overall['group'] != 'default':
        print(f"   Group: {best_overall['group']}")
    print(f"   Epoch: {best_overall['epoch']}")
    print(f"   TLCC Threshold: {best_overall['tlcc_threshold']}")
    print(f"   F1 Score: {best_overall['best_f1']:.4f}")
    print(f"   Method: {best_overall['best_method'].upper()}")
    
    # 데이터셋별 최고 성능
    print(f"\n📊 데이터셋별 최고 성능:")
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        best_dataset = dataset_data.loc[dataset_data['best_f1'].idxmax()]
        
        print(f"\n   {dataset}:")
        if best_dataset['group'] != 'default':
            print(f"     Group: {best_dataset['group']}")
        print(f"     Epoch: {best_dataset['epoch']}")
        print(f"     TLCC Threshold: {best_dataset['tlcc_threshold']}")
        print(f"     F1 Score: {best_dataset['best_f1']:.4f}")
        print(f"     ROC-AUC: {best_dataset['epsilon_roc_auc']:.4f}")
        print(f"     PR-AUC: {best_dataset['epsilon_pr_auc']:.4f}")
        print(f"     MCC: {best_dataset['epsilon_mcc']:.4f}")

def generate_research_summary(df, output_file='research_summary.txt'):
    """연구 요약 리포트 생성"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("이상 탐지 포괄적 실험 결과 요약 리포트\n")
        f.write("="*80 + "\n\n")
        
        # 실험 개요
        f.write("📋 실험 개요\n")
        f.write("-" * 40 + "\n")
        f.write(f"총 실험 수: {len(df)}\n")
        f.write(f"데이터셋: {', '.join(df['dataset'].unique())}\n")
        f.write(f"에폭 범위: {df['epoch'].min()} - {df['epoch'].max()}\n")
        f.write(f"TLCC Threshold 범위: {df['tlcc_threshold'].min()} - {df['tlcc_threshold'].max()}\n\n")
        
        # 주요 발견사항
        f.write("🔍 주요 발견사항\n")
        f.write("-" * 40 + "\n")
        
        # 최고 성능 TLCC threshold
        best_tlcc = df.groupby('tlcc_threshold')['best_f1'].mean().idxmax()
        f.write(f"1. 최적 TLCC Threshold: {best_tlcc}\n")
        
        # 데이터셋별 평균 성능
        f.write("2. 데이터셋별 평균 F1 성능:\n")
        for dataset in df['dataset'].unique():
            avg_f1 = df[df['dataset'] == dataset]['best_f1'].mean()
            f.write(f"   - {dataset}: {avg_f1:.4f}\n")
        
        # 에폭 효과
        epoch_corr = df['epoch'].corr(df['best_f1'])
        f.write(f"3. 에폭과 성능 상관관계: {epoch_corr:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"📄 연구 요약 리포트 생성: {output_file}")

def main():
    """메인 분석 함수"""
    print("📊 포괄적 실험 결과 분석 시작")
    
    # 최신 결과 파일 찾기 또는 파일명 지정
    import glob
    result_files = glob.glob("comprehensive_experiment_results_*.csv")
    if result_files:
        latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"📁 분석 대상 파일: {latest_file}")
    else:
        print("❌ 실험 결과 파일을 찾을 수 없습니다.")
        return
    
    # 데이터 로드
    df = load_experiment_results(latest_file)
    if df is None:
        return
    
    # 분석 실행
    analyze_by_dataset(df)
    analyze_by_tlcc_threshold(df)
    analyze_by_epoch(df)
    find_best_configurations(df)
    
    # 시각화 생성
    create_performance_visualizations(df)
    create_heatmap_analysis(df)
    
    # 연구 요약 생성
    generate_research_summary(df)
    
    print("\n🎉 분석 완료!")

if __name__ == "__main__":
    main()
