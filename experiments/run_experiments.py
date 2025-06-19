#!/usr/bin/env python3
"""
실험 실행 마스터 스크립트
"""

import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """명령어 실행 및 결과 처리"""
    print(f"\n🚀 {description}")
    print(f"📝 실행: {cmd}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 완료 ({duration:.1f}초)")
            return True
        else:
            print(f"❌ 실패 ({duration:.1f}초)")
            print(f"오류: {result.stderr[:500]}...")
            return False
    except Exception as e:
        print(f"❌ 예외 발생: {str(e)}")
        return False

def main():
    """실험 마스터 실행"""
    print("🎯 이상 탐지 실험 마스터 스크립트")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 작업 디렉토리 확인
    import os
    os.chdir('/home/timeseries/[filtered_data]S53CCF/Smartfactory/experiment_WaDi')
    print(f"📂 작업 디렉토리: {os.getcwd()}")
    
    # 실험 단계별 실행
    experiments = [
        {
            "cmd": "python comprehensive_experiment.py",
            "desc": "포괄적 실험 실행 (32개 실험, 예상 2.9시간)",
            "critical": True
        },
        {
            "cmd": "python analyze_comprehensive_results.py",
            "desc": "실험 결과 분석 및 시각화",
            "critical": True
        },
        {
            "cmd": "python paper_analysis.py",
            "desc": "논문용 고급 분석 및 LaTeX 테이블 생성",
            "critical": False
        }
    ]
    
    success_count = 0
    total_start_time = time.time()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n🔄 단계 {i}/{len(experiments)}: {exp['desc']}")
        print("-" * 50)
        
        success = run_command(exp['cmd'], exp['desc'])
        
        if success:
            success_count += 1
            print(f"✅ 단계 {i} 성공")
        else:
            print(f"❌ 단계 {i} 실패")
            if exp['critical']:
                print("🚨 중요한 단계가 실패했습니다. 실험을 중단합니다.")
                break
            else:
                print("⚠️ 선택적 단계입니다. 계속 진행합니다.")
        
        # 진행 상황 요약
        elapsed_time = time.time() - total_start_time
        print(f"📊 현재까지 진행 시간: {elapsed_time/60:.1f}분")
    
    # 최종 요약
    total_duration = time.time() - total_start_time
    print(f"\n" + "="*60)
    print(f"🎉 실험 완료!")
    print(f"📊 성공한 단계: {success_count}/{len(experiments)}")
    print(f"⏰ 총 소요 시간: {total_duration/60:.1f}분 ({total_duration/3600:.1f}시간)")
    print(f"📅 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 결과 파일 확인
    import glob
    result_files = glob.glob("comprehensive_experiment_results_*.csv")
    if result_files:
        latest_file = max(result_files, key=lambda x: os.path.getmtime(x))
        print(f"📁 결과 파일: {latest_file}")
        
        # 간단한 결과 요약
        try:
            import pandas as pd
            df = pd.read_csv(latest_file)
            print(f"📊 총 실험 수: {len(df)}")
            print(f"📈 평균 F1 점수: {df['best_f1'].mean():.4f}")
            print(f"🏆 최고 F1 점수: {df['best_f1'].max():.4f}")
        except:
            print("📊 결과 요약 생성 실패")
    
    print("="*60)

if __name__ == "__main__":
    main()
