# 초기 100사이클 기반 배터리 총수명 예측

초기 100사이클의 전압별 방전 용량 변화, 용량, 내부저항, 온도, 충전시간과 충전 조건으로 배터리 셀의 총수명 `cycle_life`를 예측합니다.

이 저장소는 같은 공통 실험 코드를 두 방식으로 실행합니다.

- `python main.py`: 전체 실험을 실행하고 터미널에 표를 출력한 뒤 그래프를 표시합니다.
- `notebooks/01_battery_modeling.ipynb`: 데이터 준비부터 평가까지 각 단계를 셀별로 실행하고 바로 결과를 확인합니다.

두 방식 모두 `src.experiment.ExperimentRunner`의 공통 단계 코드를 사용합니다. `main.py`는 `run_experiment(CONFIG)`로 모든 단계를 연속 실행하고, 노트북은 같은 단계를 셀별로 호출합니다. 학습 코드를 두 실행 방식에 따로 작성하지 않습니다.

## 프로젝트 구조

```text
DS-MINI/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── main.py
├── data/
│   ├── README.md
│   └── raw/
├── notebooks/
│   └── 01_battery_modeling.ipynb
└── src/
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── modeling.py
    ├── experiment.py
    └── reporting.py
```

## 설치

Python 3.14 환경에서 확인한 패키지 버전을 `requirements.txt`에 고정했습니다.

### macOS·Linux

```bash
git clone <저장소 주소>
cd DS-MINI

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ds-mini --display-name "Python (DS-MINI)"
```

### Windows PowerShell

```powershell
git clone <저장소 주소>
cd DS-MINI

python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ds-mini --display-name "Python (DS-MINI)"
```

`ipykernel` 등록은 한 번만 수행하면 됩니다. 이후 터미널에서 `.venv`를 활성화한 상태로 실행합니다.

## 데이터 준비

대용량 원본은 Git에 포함하지 않습니다. [data/README.md](data/README.md)의 파일 두 개를 `data/raw/`에 배치합니다.

프로그램은 다음 순서로 원본을 확인합니다.

1. `config.py`의 `data_dir` 또는 `ESS_DATA_DIR`
2. 현재 작업공간의 기존 로컬 데이터 후보 위치
3. `config.py`에 직접 다운로드 주소가 등록된 경우에만 다운로드
4. 확보하지 못하면 필요한 파일명을 안내하고 중단

개인 컴퓨터의 절대경로는 코드에 들어 있지 않습니다. Batch3와 추가 가변충전 데이터는 읽지 않습니다.

## 터미널 실행

```bash
python main.py
```

실행 과정은 다음과 같습니다.

```text
원본 읽기·품질 처리
→ 고정 Train·Valid·Batch2 분할
→ 초기 피처 계산
→ 전체 후보 1차 CV
→ 분기별 후보 선별
→ 확장·변화 피처 CV
→ 생존 후보 매개변수 탐색
→ Train 36셀 최종 학습
→ Valid 10셀·Batch2 39셀 평가
→ 단수명·프로토콜·큰 오차 분석
```

## 노트북 실행

```bash
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\activate
python -m jupyter lab
```

JupyterLab에서 `notebooks/01_battery_modeling.ipynb`를 연 뒤 다음 순서로 실행합니다.

1. 화면 오른쪽 위의 커널 이름을 선택합니다.
2. 등록한 **Python (DS-MINI)** 커널을 선택합니다.
3. **Kernel → Restart Kernel and Run All Cells**를 실행합니다.

커널 목록에 나타나지 않으면 JupyterLab을 종료하고 가상환경이 활성화되었는지 확인한 뒤 커널 등록 명령을 다시 실행합니다.

노트북은 다음 순서로 셀을 실행합니다.

```text
원본 읽기·분할 → 피처 계산 → 1차 CV → 확장 CV → 매개변수 탐색
→ 최종 학습·평가 → 오류 분석 → 그래프
```

각 단계 셀은 `ExperimentRunner`의 해당 메서드를 호출한 뒤 결과를 바로 표시합니다. 완료된 단계의 셀을 다시 실행하면 기존 상태를 반환하므로 같은 모델을 중복 학습하지 않습니다. 설정을 바꿨다면 커널을 재시작하고 첫 셀부터 다시 실행합니다.

## 공통 실험 설정

모든 설정은 `config.py`의 `ExperimentConfig`에서 관리합니다.

- 난수: 42
- Batch1 검증 셀: 10
- 학습 내부 검증: 5-fold
- 타깃: 원래 수명, `log10(cycle_life)`
- 모델: ElasticNet, Ridge, RandomForest
- 기준선: 학습 fold 평균을 사용하는 DummyRegressor
- 전처리: 학습 fold 중앙값 대체와 표준화
- 선택 기준: 학습 내부 CV 평균 MAPE
- 단계별 생존 후보 수와 모델별 탐색 범위
- 최종 평가 실행 여부

Batch1 셀 목록은 정렬한 뒤 `train_test_split(test_size=10, random_state=42)`로 한 번 분할합니다. 학습용 36셀의 5개 fold도 한 번 만들고 모든 후보가 재사용합니다.

## 피처 후보

초기 후보는 다음 세 그룹으로 구성합니다.

- 설계 피처: A, B, C-time, C-capacity, C-resistance, C-temperature, C-policy
- 기존 조합: 용량과 충전 조건·온도·내부저항 조합
- 변경 조합: 평균 Qd 제외, 상대 용량 변화, 정규화 ΔQ, 온도·내부저항 변화

1차 CV의 모델·타깃·피처 그룹별 상위 후보에서 변화 피처 추가, 기울기 교체, 평균 Qd 제외와 정규화 ΔQ 후보를 확장합니다. 생존 후보에만 모델별 매개변수 탐색을 적용합니다.

물리량 기반 피처는 표준화 전에 계산합니다. 결측 대체와 표준화는 각 CV 학습 fold에서만 학습하며 Valid와 Batch2에는 변환만 적용합니다.

## 데이터 처리 기준

- 실제 사이클 번호 100 이하만 사용합니다.
- ΔQ는 같은 전압격자의 `Q100 - Q10`입니다.
- ΔQ 로그 분산은 `ddof=0`, 하한 `1e-12`를 사용합니다.
- 방전 용량 평균·기울기는 2∼100사이클에서 계산합니다.
- 상대 용량 변화는 `(91∼100 평균 - 2∼11 평균) / 2∼11 평균`입니다.
- 내부저항·온도 변화는 뒤 구간 평균에서 앞 구간 평균을 뺍니다.
- 정규화 ΔQ는 셀의 2∼11사이클 평균 Qd로 나눕니다.
- Batch1 첫 사이클의 확인된 빈 온도·충전시간은 결측 처리합니다.
- 내부저항 0은 결측 처리합니다.
- 확인된 Batch2 `cell038`에서 `Tmin=-270` 또는 `Tmax=400`인 행의 온도 요약값만 결측 처리하고 셀은 유지합니다.
- 충전 조건 해석에 실패하면 원문은 보존하고 숫자 피처는 결측으로 둡니다.
- 짧은 수명을 이유로 셀을 삭제하거나 라벨을 바꾸지 않습니다.

## 결과 해석 범위

실행 중 계산된 표·예측·그래프는 메모리에만 존재합니다. 별도 결과 CSV, 모델, 그래프 파일은 저장하지 않습니다.

후보 탐색 자체는 설정된 전체 조합으로 수행하지만, 터미널과 노트북에는 단계별로 최대 10개만 표시합니다. 표시 후보는 CV 성능을 우선하면서 모델·타깃 변환·피처군·기존/변경 구성의 다양성을 함께 반영하고, 전체 평가 수와 생존·탈락 사유 집계는 별도로 유지합니다.

Batch2를 이미 확인한 후속 탐색이므로 실행 결과는 탐색 실험입니다. Batch2 점수로 후보를 선택하지 않으며, 9.1%는 과제의 비교 목표로만 사용합니다. 논문과 동일한 분할의 재현 성능으로 표현하지 않습니다.

## Git 제출 전 점검

- `data/raw/` 원본이 Git 추적 대상에 포함되지 않았는지 확인합니다.
- 새 가상환경에서 의존 패키지가 설치되는지 확인합니다.
- `python main.py`와 노트북 전체 실행의 분할·후보·최종 구성·예측·지표가 일치하는지 확인합니다.
- 코드와 노트북에 개인 절대경로가 없는지 확인합니다.
- Batch3가 처리 대상에 들어가지 않는지 확인합니다.
- 실행 결과 파일을 생성하지 않는지 확인합니다.

현재 폴더는 `main` 브랜치의 독립 Git 저장소로 초기화되어 있습니다. 점검 후 다음 순서로 새 GitHub 저장소에 게시할 수 있습니다.

```bash
git add .
git status
git commit -m "Implement battery cycle-life modeling workflow"
git remote add origin https://github.com/<계정>/<저장소>.git
git push -u origin main
```

커밋 전 `git status`에 `data/raw/`의 원본 파일이 나타나지 않는지 다시 확인합니다.
