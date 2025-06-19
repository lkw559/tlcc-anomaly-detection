#!/usr/bin/env python3
"""
실험 모니터링 및 상태 확인 스크립트
"""

import os
import time
import glob
from datetime import datetime
from pathlib import Path

def monitor_experiments():
    """실험 진행 상황 모니터링"""
    print("📊 실험 모니터링 시작")
    print("Press Ctrl+C to stop monitoring")
    
    output_base = "/home/timeseries/[filtered_data]S53CCF/Smartfactory/experiment_WaDi/output"
    
    last_count = 0
    
    try:
        while True:
            # 현재 실험 폴더 수 계산
            current_count = 0
            recent_folders = []
            
            for dataset in ['WADI', 'SMAP', 'MSL', 'SMD']:
                dataset_path = os.path.join(output_base, dataset)
                if os.path.exists(dataset_path):
                    if dataset == 'SMD':
                        # SMD는 하위 그룹 폴더가 있음
                        for group_folder in os.listdir(dataset_path):
                            group_path = os.path.join(dataset_path, group_folder)
                            if os.path.isdir(group_path):
                                exp_folders = [f for f in os.listdir(group_path) 
                                             if f.startswith('20') and os.path.isdir(os.path.join(group_path, f))]
                                current_count += len(exp_folders)
                                
                                # 최근 폴더 찾기
                                for folder in exp_folders:
                                    folder_path = os.path.join(group_path, folder)
                                    mtime = os.path.getmtime(folder_path)
                                    if time.time() - mtime < 300:  # 5분 이내
                                        recent_folders.append(f"{dataset}/{group_folder}/{folder}")
                    else:
                        # 다른 데이터셋
                        exp_folders = [f for f in os.listdir(dataset_path) 
                                     if f.startswith('20') and os.path.isdir(os.path.join(dataset_path, f))]
                        current_count += len(exp_folders)
                        
                        # 최근 폴더 찾기
                        for folder in exp_folders:
                            folder_path = os.path.join(dataset_path, folder)
                            mtime = os.path.getmtime(folder_path)
                            if time.time() - mtime < 300:  # 5분 이내
                                recent_folders.append(f"{dataset}/{folder}")
            
            # 상태 출력
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\n⏰ {current_time} | 총 실험 폴더: {current_count}")
            
            if current_count > last_count:
                new_experiments = current_count - last_count
                print(f"✅ 새로 완료된 실험: {new_experiments}개")
                last_count = current_count
            
            if recent_folders:
                print("🔥 최근 실험 폴더:")
                for folder in recent_folders[-5:]:  # 최근 5개만 표시
                    print(f"   {folder}")
            
            # GPU 사용량 확인 (선택적)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip()
                    print(f"🖥️  GPU: {gpu_info}")
            except:
                pass
            
            time.sleep(30)  # 30초마다 확인
            
    except KeyboardInterrupt:
        print("\n📊 모니터링 종료")

def check_latest_results():
    """최신 실험 결과 확인"""
    print("📋 최신 실험 결과 확인")
    
    output_base = "/home/timeseries/[filtered_data]S53CCF/Smartfactory/experiment_WaDi/output"
    
    latest_results = []
    
    for dataset in ['WADI', 'SMAP', 'MSL', 'SMD']:
        dataset_path = os.path.join(output_base, dataset)
        if os.path.exists(dataset_path):
            if dataset == 'SMD':
                # SMD 그룹별 확인
                for group_folder in os.listdir(dataset_path):
                    group_path = os.path.join(dataset_path, group_folder)
                    if os.path.isdir(group_path):
                        exp_folders = [(f, os.path.getmtime(os.path.join(group_path, f))) 
                                     for f in os.listdir(group_path) 
                                     if f.startswith('20') and os.path.isdir(os.path.join(group_path, f))]
                        
                        if exp_folders:
                            latest_folder = max(exp_folders, key=lambda x: x[1])
                            summary_path = os.path.join(group_path, latest_folder[0], 'summary.txt')
                            if os.path.exists(summary_path):
                                latest_results.append((f"{dataset}/{group_folder}/{latest_folder[0]}", 
                                                     latest_folder[1], summary_path))
            else:
                exp_folders = [(f, os.path.getmtime(os.path.join(dataset_path, f))) 
                             for f in os.listdir(dataset_path) 
                             if f.startswith('20') and os.path.isdir(os.path.join(dataset_path, f))]
                
                if exp_folders:
                    latest_folder = max(exp_folders, key=lambda x: x[1])
                    summary_path = os.path.join(dataset_path, latest_folder[0], 'summary.txt')
                    if os.path.exists(summary_path):
                        latest_results.append((f"{dataset}/{latest_folder[0]}", 
                                             latest_folder[1], summary_path))
    
    # 시간순 정렬
    latest_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 최신 실험 결과 ({len(latest_results)}개):")
    
    for i, (folder, mtime, summary_path) in enumerate(latest_results[:10]):  # 최신 10개만
        try:
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # 최고 성능 추출
            epsilon_f1 = summary.get('epsilon_extended', {}).get('f1', 0)
            pot_f1 = summary.get('pot_extended', {}).get('f1', 0)
            best_f1 = max(epsilon_f1, pot_f1)
            best_method = 'Epsilon' if epsilon_f1 >= pot_f1 else 'POT'
            
            # 추가 지표
            epsilon_ext = summary.get('epsilon_extended', {})
            roc_auc = epsilon_ext.get('roc_auc', 0)
            pr_auc = epsilon_ext.get('pr_auc', 0)
            mcc = epsilon_ext.get('mcc', 0)
            
            time_str = datetime.fromtimestamp(mtime).strftime('%m-%d %H:%M')
            
            print(f"  {i+1:2d}. {folder}")
            print(f"      시간: {time_str} | 최고 F1: {best_f1:.4f} ({best_method})")
            print(f"      ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | MCC: {mcc:.4f}")
            
        except Exception as e:
            print(f"  {i+1:2d}. {folder} (결과 읽기 실패: {str(e)})")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'monitor':
        monitor_experiments()
    else:
        check_latest_results()
