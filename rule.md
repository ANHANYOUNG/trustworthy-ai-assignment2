# Assignment 2 Rules

기준 문서: `assignment2.pdf`

## 핵심

- 주제: `Differential Testing with DeepXplore`
- 대상 모델: `CIFAR-10`에 대해 학습된 `ResNet50` 최소 2개
- 핵심 개념:
  - 모델 간 `prediction disagreement` 탐지
  - `neuron coverage` 측정
- 실행 파일: `test.py`
- 최종 제출물: `requirements.txt`, `test.py`, `results/`, `report.pdf`, `README.md`

## 일정 / 제출

- 배포일: `Wednesday, April 8, 2026`
- 마감일: `Wednesday, April 22, 2026, 11:59 PM`
- 제출 방식: GitHub 업로드 후 Uclass 링크 제출
- 지각 제출: 시스템 / 이메일 모두 불가

## 프로젝트 구성

- `requirements.txt` 포함
- 외부 모듈 / Python dependency 전부 기록
- 코드 이해용 주석 포함
- `test.py` 포함
- `test.py` 실행 시 DeepXplore를 모델들에 대해 실행하고 결과를 보여줄 수 있어야 함
- 테스트 형식 자유, 기능과 타당성 확인 가능 상태

## Problem 1: Differential Testing with DeepXplore

### 배경

- DeepXplore는 동일 입력을 여러 신경망에 넣고 `prediction disagreement`를 찾는 `differential testing` 방식
- 핵심 아이디어:
  - independently trained models가 같은 입력에 대해 다른 출력을 내면 적어도 하나는 틀렸을 가능성 존재
- 추가 목표:
  - `neuron coverage` 최대화
- `neuron coverage`:
  - threshold 이상 활성화된 뉴런의 비율

### 수행 내용

#### 1. DeepXplore 설정

- DeepXplore repository clone
- 필요 dependency 설치
- 원본 구현은 오래된 library version 기반일 수 있으므로 현재 framework에 맞는 수정 필요 가능성 존재

#### 2. 모델 준비

- `CIFAR-10`용 `ResNet50` 최소 2개 준비
- 직접 학습 가능
- pre-trained model 사용 가능
- pre-trained model 사용 시 서로 다른 initialization, hyperparameter, training procedure 등 차이 필요
- open-source model 사용 시 report에 proper citation 필요

#### 3. 실행

- DeepXplore를 CIFAR-10 ResNet50 모델들에 맞게 설정
- test generation process 실행
- 모델 간 disagreement가 발생하는 입력 수집

#### 4. 주의 사항

- 원본 DeepXplore 구현은 `CIFAR-10` 및 `ResNet50`에 맞게 수정 필요 가능성 존재
- 특히 input preprocessing 확인 필요:
  - `normalization`
  - `resizing`
- 모델과 DeepXplore 간 preprocessing consistency 유지 필요
- 수정 사항 문서화 필요

## Problem 2: Connecting Attacks and Testing

### 배경

- Assignment #1에서는 `FGSM`, `PGD` adversarial attack 구현
- DeepXplore 같은 testing tool 역시 neural network weakness를 찾는 목적
- 두 접근의 공통 목표:
  - 모델의 weakness uncovering

### 에세이 작성

- 분량: `500 words` 이하
- 주제:
  - Assignment #1의 attack methods와 DeepXplore 같은 testing tool을 결합하여 testing 또는 attack을 가속 / 강화할 수 있는지 논의

### Points to consider

1. `Synergies`
   - adversarial attacks가 differential testing의 seed input이 될 수 있는가
   - neuron coverage metric이 perturbation target 선택에 도움을 줄 수 있는가

2. `Coverage-guided attacks`
   - gradient-based attacks를 misclassification 대신 또는 추가로 neuron coverage 최대화 방향으로 바꿀 수 있는가

3. `Efficiency`
   - 두 기법 결합이 단독 사용보다 더 빠르게 더 많은 bugs를 찾게 할 수 있는가

4. `Limitations`
   - 두 접근 결합이 도움이 되지 않는 상황이 있는가

## 채점 / 제출 결과

### Problem 1: DeepXplore (`60%`)

#### 1. Code & reproducibility (`30%`)

- `test.py` 실행 가능
- clear setup instructions 포함

#### 2. Results & analysis (`30%`)

- 보고 항목:
  - disagreement-inducing inputs 개수
  - neuron coverage
  - 최소 `5개`의 suspicious input 시각화
  - 시각화는 각 모델 prediction과 함께 `results/` directory에 저장
  - 어떤 입력이 disagreement를 유발하는지와 그 이유에 대한 짧은 논의

### Problem 2: Essay (`40%`)

- `Clarity and logic`: `20%`
- `Depth of analysis`: `20%`

## AI 사용 / Git 히스토리

- AI 도구 사용 가능
- 전제: genuine understanding 필요
- 저장소 요구:
  - meaningful commit history
  - single bulk commit 지양
  - git log inspect 가능성 고려
- 보고서 요구:
  - own interpretation and reasoning
  - generic description 지양
  - specific observations and connections 포함

## 최종 제출물

- `requirements.txt`
- `test.py`
- `results/`
- `report.pdf`
- `README.md`

## README 요구사항

- setup 및 run instructions 포함
- DeepXplore에 가한 modifications 설명 포함