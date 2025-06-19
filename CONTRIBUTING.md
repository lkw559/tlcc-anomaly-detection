# 기여 가이드

TLCC 기반 이상 탐지 프로젝트에 기여해주셔서 감사합니다! 이 문서는 프로젝트에 기여하는 방법을 안내합니다.

## 🚀 시작하기

### 1. 저장소 포크
GitHub에서 이 저장소를 포크하고 로컬에 클론합니다:

```bash
git clone https://github.com/your-username/tlcc-anomaly-detection.git
cd tlcc-anomaly-detection
```

### 2. 개발 환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .  # 개발 모드로 패키지 설치
```

### 3. 개발 의존성 설치
```bash
pip install black isort flake8 pytest
```

## 📝 개발 프로세스

### 브랜치 전략
- `main`: 안정적인 릴리스 브랜치
- `develop`: 개발 브랜치
- `feature/feature-name`: 새로운 기능 개발
- `bugfix/bug-description`: 버그 수정
- `hotfix/critical-fix`: 긴급 수정

### 새로운 기능 추가
1. `develop` 브랜치에서 새 브랜치 생성:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. 기능 개발 및 테스트 작성

3. 커밋 메시지 작성 (아래 규칙 참고)

4. Pull Request 생성

## 📐 코딩 스타일

### Python 스타일 가이드
- [PEP 8](https://www.python.org/dev/peps/pep-0008/) 준수
- 최대 줄 길이: 88자 (Black 기본값)
- Type hints 사용 권장

### 코드 포매팅
```bash
# 코드 포매팅
black .

# 임포트 정렬
isort .

# 린팅
flake8 .
```

### 문서화
- 모든 함수와 클래스에 docstring 작성
- 복잡한 알고리즘에는 주석 추가
- README 업데이트 (필요시)

## 🧪 테스팅

### 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/

# 특정 테스트 파일 실행
pytest tests/test_evaluation.py

# 커버리지 확인
pytest --cov=src tests/
```

### 새 테스트 작성
- 새로운 기능에는 반드시 테스트 추가
- `tests/` 디렉토리에 `test_*.py` 파일 생성
- unittest 또는 pytest 프레임워크 사용

## 📋 커밋 메시지 규칙

### 형식
```
type(scope): subject

body

footer
```

### 타입
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포매팅 (기능 변경 없음)
- `refactor`: 코드 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드 프로세스나 도구 변경

### 예시
```
feat(evaluation): add ROC-AUC and PR-AUC metrics

- ROC-AUC 계산 함수 추가
- PR-AUC 계산 함수 추가
- 확장된 평가 지표 표시 함수 개선

Closes #123
```

## 🐛 버그 리포트

버그를 발견했다면 다음 정보와 함께 이슈를 생성해주세요:

### 필수 정보
- **OS**: (예: Ubuntu 20.04)
- **Python 버전**: (예: 3.8.5)
- **PyTorch 버전**: (예: 1.9.0)
- **오류 메시지**: 전체 스택 트레이스
- **재현 단계**: 오류를 재현하는 단계별 설명

### 템플릿
```markdown
## 버그 설명
간단한 버그 설명

## 재현 단계
1. 첫 번째 단계
2. 두 번째 단계
3. 오류 발생

## 예상 동작
어떤 동작을 예상했는지

## 실제 동작
실제로 어떤 일이 일어났는지

## 환경
- OS: 
- Python: 
- PyTorch: 
- 기타 관련 패키지:

## 추가 정보
스크린샷, 로그 파일 등
```

## 💡 기능 제안

새로운 기능을 제안할 때는 다음을 고려해주세요:

### 제안서 템플릿
```markdown
## 기능 설명
제안하는 기능에 대한 명확한 설명

## 동기
이 기능이 왜 필요한지, 어떤 문제를 해결하는지

## 상세 설계
구현 방법에 대한 아이디어

## 대안
고려한 다른 해결 방법들

## 추가 정보
관련 논문, 참고 자료 등
```

## 📚 문서 기여

문서 개선도 환영합니다:

- README.md 개선
- 코드 주석 추가
- 예제 코드 작성
- 튜토리얼 작성

## 🔄 Pull Request 가이드

### PR 체크리스트
- [ ] 관련 이슈에 연결됨
- [ ] 테스트가 추가/업데이트됨
- [ ] 문서가 업데이트됨 (필요시)
- [ ] 코드 스타일 가이드 준수
- [ ] 모든 테스트 통과
- [ ] 상세한 PR 설명 작성

### PR 템플릿
```markdown
## 변경 사항
이 PR에서 수행한 변경 사항 설명

## 관련 이슈
Fixes #(이슈 번호)

## 테스트
수행한 테스트에 대한 설명

## 체크리스트
- [ ] 코드가 스타일 가이드를 따름
- [ ] 테스트 추가/업데이트
- [ ] 문서 업데이트 (필요시)
```

## 👥 코드 리뷰

### 리뷰어 가이드
- 건설적인 피드백 제공
- 코드 품질, 성능, 보안 검토
- 문서화 및 테스트 확인

### 작성자 가이드
- 피드백에 적극적으로 응답
- 필요시 코드 수정
- 리뷰어에게 감사 표현

## 📞 도움이 필요한 경우

- GitHub 이슈로 질문 생성
- 디스커션 탭 활용
- 이메일: your.email@example.com

## 🎉 인정

모든 기여자의 이름은 CONTRIBUTORS.md 파일에 기록됩니다.

다시 한 번, 프로젝트에 기여해주셔서 감사합니다! 🙏
