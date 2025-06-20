#!/bin/bash

# GitHub 저장소 업로드 스크립트
# 사용법: ./upload_commands.sh

echo "=== TLCC Anomaly Detection 프로젝트 GitHub 업로드 ==="
echo "저장소 URL: https://github.com/lkw559/tlcc-anomaly-detection"
echo ""

# Git 사용자 설정 (필요시)
echo "Git 사용자 설정..."
git config --global user.name "lkw559"
git config --global user.email "lkw559@example.com"  # 실제 이메일로 변경

# 원격 저장소 재설정
echo "원격 저장소 설정..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/lkw559/tlcc-anomaly-detection.git

# 브랜치 설정
git branch -M main

# 업로드 실행
echo "GitHub에 업로드 중..."
git push -u origin main

echo ""
echo "✅ 업로드 완료!"
echo "저장소 확인: https://github.com/lkw559/tlcc-anomaly-detection"
