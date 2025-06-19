#!/usr/bin/env python3
"""
포괄적 이상 탐지 실험 스크립트 - 효율적 버전
- 4개 데이터셋 (WADI, SMAP, MSL, SMD-1-1)
- 2개 에폭 (3, 5)
- 4개 TLCC threshold (0.0, 0.3, 0.5, 0.7)
- 총 32개 실험, 약 2.9시간 예상
"""

import os
import sys
import json
import pandas as pd
import subprocess
import time
from datetime import datetime
from pathlib import Path

# 실험 설정
DATASETS = {
    'WADI': {},
    'SMAP': {},
    'MSL': {},
    'SMD': {'groups': ['1-1']}
}

EPOCHS = [3, 5]
TLCC_THRESHOLDS = [0.0, 0.3, 0.5, 0.7]
BATCH_SIZE = 128

# 결과 저장용 컬럼
results_columns = [
    'timestamp', 'dataset', 'group', 'epoch', 'tlcc_threshold', 
    'use_true_tlcc', 'tlcc_binary',
    # Epsilon Extended Metrics
    'epsilon_f1', 'epsilon_precision', 'epsilon_recall',
    'epsilon_roc_auc', 'epsilon_pr_auc', 'epsilon_mcc',
    'epsilon_tp', 'epsilon_tn', 'epsilon_fp', 'epsilon_fn',
    'epsilon_threshold', 'epsilon_latency',
    # POT Extended Metrics  
    'pot_f1', 'pot_precision', 'pot_recall',
    'pot_roc_auc', 'pot_pr_auc', 'pot_mcc',
    'pot_tp', 'pot_tn', 'pot_fp', 'pot_fn',
    'pot_threshold', 'pot_latency',
    # Best Method
    'best_method', 'best_f1',
    # Experiment Info
    'output_path', 'total_time'
]

def create_experiment_dataframe():
    """실험 결과 저장용 데이터프레임 생성"""
    return pd.DataFrame(columns=results_columns)

def run_single_experiment(dataset, group, epoch, tlcc_threshold, results_df):
    """단일 실험 실행"""
    print(f"\n{'='*80}")
    print(f"🧪 실험: {dataset} | Group: {group} | Epoch: {epoch} | TLCC: {tlcc_threshold}")
    print(f"{'='*80}")
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 실험 명령어 구성
    comment = f"{dataset}_comprehensive_tlcc_{tlcc_threshold}_epoch_{epoch}"
    if group:
        comment += f"_group_{group}"
    
    cmd = [
        'python', 'train_original.py',
        '--comment', comment,
        '--epoch', str(epoch),
        '--bs', str(BATCH_SIZE),
        '--dataset', dataset,
        '--tlcc_threshold', str(tlcc_threshold),
        '--use_true_tlcc', '1',
        '--tlcc_binary', '1'
    ]
    
    if dataset == 'SMD' and group:
        cmd.extend(['--group', group])
    
    try:
        print(f"🚀 실행: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"❌ 실험 실패: {result.stderr[:500]}...")
            return results_df
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 결과 파일 찾기
        output_pattern = f"output/{dataset}"
        if group:
            output_pattern += f"/{group}"
        
        # 가장 최신 폴더 찾기 (더 안정적인 방법)
        import glob
        pattern = f"{output_pattern}/*"
        output_dirs = glob.glob(pattern)
        
        # 디렉토리만 필터링하고 최신순 정렬
        output_dirs = [d for d in output_dirs if os.path.isdir(d) and os.path.basename(d).startswith('20')]
        
        if not output_dirs:
            print(f"❌ 결과 폴더를 찾을 수 없음: {output_pattern}")
            return results_df
        
        # 가장 최신 폴더 선택 (수정 시간 기준)
        latest_output = max(output_dirs, key=os.path.getmtime)
        summary_file = os.path.join(latest_output, 'summary.txt')
        
        if not os.path.exists(summary_file):
            print(f"❌ Summary 파일 없음: {summary_file}")
            return results_df
        
        # Summary 파일 읽기
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # 결과 추출
        epsilon_ext = summary.get('epsilon_extended', {})
        pot_ext = summary.get('pot_extended', {})
        
        # 최고 성능 방법 결정
        epsilon_f1 = epsilon_ext.get('f1', 0)
        pot_f1 = pot_ext.get('f1', 0)
        
        if epsilon_f1 >= pot_f1:
            best_method = 'epsilon'
            best_f1 = epsilon_f1
        else:
            best_method = 'pot'
            best_f1 = pot_f1
        
        # 결과 행 생성
        result_row = {
            'timestamp': timestamp,
            'dataset': dataset,
            'group': group if group else 'default',
            'epoch': epoch,
            'tlcc_threshold': tlcc_threshold,
            'use_true_tlcc': 1,
            'tlcc_binary': 1,
            # Epsilon Extended
            'epsilon_f1': epsilon_ext.get('f1', 0),
            'epsilon_precision': epsilon_ext.get('precision', 0),
            'epsilon_recall': epsilon_ext.get('recall', 0),
            'epsilon_roc_auc': epsilon_ext.get('roc_auc', 0),
            'epsilon_pr_auc': epsilon_ext.get('pr_auc', 0),
            'epsilon_mcc': epsilon_ext.get('mcc', 0),
            'epsilon_tp': epsilon_ext.get('TP', 0),
            'epsilon_tn': epsilon_ext.get('TN', 0),
            'epsilon_fp': epsilon_ext.get('FP', 0),
            'epsilon_fn': epsilon_ext.get('FN', 0),
            'epsilon_threshold': epsilon_ext.get('threshold', 0),
            'epsilon_latency': epsilon_ext.get('latency', 0),
            # POT Extended
            'pot_f1': pot_ext.get('f1', 0),
            'pot_precision': pot_ext.get('precision', 0),
            'pot_recall': pot_ext.get('recall', 0),
            'pot_roc_auc': pot_ext.get('roc_auc', 0),
            'pot_pr_auc': pot_ext.get('pr_auc', 0),
            'pot_mcc': pot_ext.get('mcc', 0),
            'pot_tp': pot_ext.get('TP', 0),
            'pot_tn': pot_ext.get('TN', 0),
            'pot_fp': pot_ext.get('FP', 0),
            'pot_fn': pot_ext.get('FN', 0),
            'pot_threshold': pot_ext.get('threshold', 0),
            'pot_latency': pot_ext.get('latency', 0),
            # Best Method
            'best_method': best_method,
            'best_f1': best_f1,
            # Experiment Info
            'output_path': latest_output,
            'total_time': total_time
        }
        
        # 데이터프레임에 추가
        results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
        
        print(f"✅ 실험 완료 ({total_time:.1f}초)")
        print(f"📊 최고 성능: {best_method.upper()} (F1: {best_f1:.4f})")
        
        # 중간 저장
        results_df.to_csv('comprehensive_experiment_results_temp.csv', index=False)
        
    except subprocess.TimeoutExpired:
        print(f"⏰ 실험 타임아웃 (1시간 초과)")
    except Exception as e:
        print(f"❌ 실험 중 오류: {str(e)}")
    
    return results_df

def run_comprehensive_experiments():
    """포괄적 실험 실행"""
    print("🎯 포괄적 이상 탐지 실험 시작")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results_df = create_experiment_dataframe()
    
    total_experiments = 0
    completed_experiments = 0
    
    # 총 실험 수 계산
    for dataset, config in DATASETS.items():
        if dataset == 'SMD':
            total_experiments += len(config['groups']) * len(EPOCHS) * len(TLCC_THRESHOLDS)
        else:
            total_experiments += len(EPOCHS) * len(TLCC_THRESHOLDS)
    
    print(f"📊 총 실험 수: {total_experiments}")
    
    # 각 데이터셋별 실험
    for dataset, config in DATASETS.items():
        print(f"\n🔬 {dataset} 데이터셋 실험 시작")
        
        if dataset == 'SMD':
            for group in config['groups']:
                for epoch in EPOCHS:
                    for tlcc_threshold in TLCC_THRESHOLDS:
                        results_df = run_single_experiment(
                            dataset, group, epoch, tlcc_threshold, results_df
                        )
                        completed_experiments += 1
                        print(f"📈 진행률: {completed_experiments}/{total_experiments} "
                              f"({completed_experiments/total_experiments*100:.1f}%)")
        else:
            for epoch in EPOCHS:
                for tlcc_threshold in TLCC_THRESHOLDS:
                    results_df = run_single_experiment(
                        dataset, None, epoch, tlcc_threshold, results_df
                    )
                    completed_experiments += 1
                    print(f"📈 진행률: {completed_experiments}/{total_experiments} "
                          f"({completed_experiments/total_experiments*100:.1f}%)")
    
    # 최종 결과 저장
    final_filename = f"comprehensive_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(final_filename, index=False)
    
    print(f"\n🎉 모든 실험 완료!")
    print(f"📁 결과 파일: {final_filename}")
    print(f"📊 총 완료된 실험: {completed_experiments}/{total_experiments}")
    
    # 간단한 요약 통계
    if len(results_df) > 0:
        print(f"\n📋 실험 요약:")
        for dataset in results_df['dataset'].unique():
            count = len(results_df[results_df['dataset'] == dataset])
            avg_f1 = results_df[results_df['dataset'] == dataset]['best_f1'].mean()
            print(f"   {dataset}: {count}개 실험, 평균 F1: {avg_f1:.4f}")
    
    return results_df

if __name__ == "__main__":
    # 작업 디렉토리 확인
    os.chdir('/home/timeseries/[filtered_data]S53CCF/Smartfactory/experiment_WaDi')
    
    # 실험 실행
    results = run_comprehensive_experiments()
