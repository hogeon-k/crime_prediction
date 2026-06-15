# Crime Prediction

범죄 발생 건수를 예측하기 위한 Python 프로젝트입니다. 정부 범죄/인구 CSV를 전처리하고, Linear Regression, Random Forest, XGBoost 계열 모델을 비교한 뒤 선택된 AI 모델로 CSV/XLSX 입력 파일 또는 Tkinter GUI에서 예측을 수행합니다.

## 주요 기능

- 범죄/인구 데이터 병합 및 전처리
- 표준 CSV/XLSX 업로드와 정부 원천 데이터 업로드 지원
- 연도 기준 자동 Train/Test 분리: 현재 원본 2018~2024, 전년도 feature 적용 후 Train 2019~2023 / Test 2024
- Linear Regression, 직접 구현 Random Forest, 직접 구현 XGBoost 비교
- 전년도 발생 건수, 전년도 범죄율, 지역/범죄유형 평균 feature 생성
- 2024년 실제 데이터를 기준으로 2025~2027 다년도 재귀 예측 결과 XLSX 저장
- Tkinter GUI 제공
- pytest 기반 테스트와 ruff/pylint 품질 점검 설정

## 설치

Python 3.11 이상을 권장합니다.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

개발 환경에서는 editable install을 사용할 수 있습니다.

```powershell
pip install -e ".[dev]"
```

## 실행 방법

최종 통합 GUI:

```powershell
python run_app.py
```

Deprecated legacy GUI:

```powershell
python src/run_gui.py
python src/run_excel_gui.py
```

위 개별 GUI는 이전 화면 확인용으로만 유지되며, 최종 실행 화면은 `run_app.py`입니다.

파일 예측 API:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from ai.predict import predict_recursive_from_file

predict_recursive_from_file("input.xlsx", "data", start_year=2025, end_year=2027)
```

기존 `predict_from_file(..., target_year=...)` API는 단일 목표 연도 독립 예측용으로 유지됩니다. 최종 GUI의 파일 예측 화면은 다년도 재귀 예측을 사용합니다.

단일 입력 예측:

```python
from ai.predict import predict_one

result = predict_one(2025, "서울", "절도", 9_000_000)
print(result)
```

## 모델 재학습

정부 범죄/인구 CSV 파일을 `src/data` 아래에 둔 뒤 실행합니다.

```powershell
python src/ai/train.py
```

학습 리포트에는 기존 모델 비교에 더해 다음 정보와 실험이 함께 출력됩니다.

- 원본 데이터 연도 범위, 전년도 feature 생성 후 사용 가능 연도, 제외된 연도
- 자동 Hold-out 검증: 데이터에 존재하는 최신 연도를 Test로, 그 이전 사용 가능 연도를 Train으로 사용
- RandomForest 후보군: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- XGBoost 후보군: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `reg_lambda`, `gamma`
- 각 후보 조합의 Train/Test R2, RMSE, MAE 및 Train/Test R2 gap
- 발생건수, 인구수, 범죄율 분포와 IQR 기준 이상값 수
- 지역별, 범죄유형별 발생건수 통계
- 숫자형 feature와 target 간 상관, split count 기반 importance, permutation importance
- 연도 기반 Walk-Forward 검증: 과거 연도로 학습하고 다음 연도를 검증

현재 원본 범죄 데이터 범위는 2018~2024입니다. 전년도 발생건수와 전년도 범죄율 feature를 사용하므로 기본 설정에서는 첫 연도인 2018년을 학습/검증 행에서 제외하고, 실제 사용 가능 범위는 2019~2024가 됩니다. 따라서 현재 데이터 기준 Hold-out 검증은 Train 2019~2023, Test 2024로 자동 구성됩니다.

최종 모델은 단일 Hold-out 결과가 아니라 Walk-Forward Validation 평균 성능을 기준으로 선택합니다. 평균 Validation R2가 가장 높은 모델을 우선하고, R2가 같거나 매우 유사하면 평균 RMSE, 평균 MAE가 낮은 모델을 선택합니다. 단일 Hold-out 결과는 참고용으로 함께 출력합니다.

`best_model.pkl`은 최종 선택된 모델을 전체 학습 가능 데이터로 다시 학습한 뒤 저장합니다. 현재 데이터 기준으로는 2019~2024 전체를 사용해 최종 재학습합니다.

전년도 발생건수 및 전년도 범죄율 feature 제거 실험은 엄밀한 의미의 모델 ablation이라기보다 특정 feature군 제거 실험 또는 feature importance 검증으로 표현하는 것이 더 정확합니다.

학습이 끝나면 로컬에 다음 파일이 생성됩니다.

- `models/best_model.pkl`
- `models/model_info.json`

`best_model.pkl`은 Git 저장소에 포함하지 않습니다. 공개 배포 시에는 GitHub Release asset, 사내 아티팩트 저장소, 또는 별도 다운로드 링크로 배포하세요.

## 테스트와 품질 점검

기본 테스트:

```powershell
pytest
```

통합 테스트까지 포함:

```powershell
pytest -m integration
```

Ruff:

```powershell
python -m ruff check .
```

Pylint:

```powershell
pylint src tests
```

## 모델 평가 기준

최종 모델은 `linear`, `random_forest`, `xgboost` 세 AI 모델만 비교하여 선택합니다. 선택 기준은 Walk-Forward 평균 Validation R2, 평균 RMSE, 평균 MAE 순서입니다.

| Model | Walk-Forward mean R2 | Walk-Forward mean RMSE | Walk-Forward mean MAE | Hold-out R2 | Hold-out RMSE | Hold-out MAE |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.9690 | 1273.33 | 444.60 | 0.9601 | 1627.16 | 476.60 |
| XGBoost | 0.9604 | 1431.79 | 379.96 | - | - | - |
| Random Forest | 0.9544 | 1522.16 | 452.95 | - | - | - |

학습 리포트에는 R2, RMSE, MAE, MSE, 예측 다양성, 지역별 MAE, 범죄 유형별 MAE를 포함합니다.

## 입력 파일 형식

예측 입력 파일은 다음 컬럼을 포함해야 합니다.

- `연도`
- `지역`
- `범죄_유형`
- `인구수`

다년도 재귀 예측 결과 파일은 다음 컬럼을 포함합니다.

- `입력_기준_연도`
- `예측_대상_연도`
- `지역`
- `범죄_유형`
- `연도`
- `인구수`
- `전년도_발생_건수`
- `전년도_범죄율`
- `예측_발생_건수`
- `예측_범죄율`
- `예측_단계`
- `예측_방식`
- `인구수_추정_방법`

2025년은 2024년 실제 `발생_건수`와 `범죄율`을 전년도 feature로 사용합니다. 2026년은 2025년 예측값을, 2027년은 2026년 예측값을 각각 다음 연도의 전년도 feature로 연결합니다.

미래 연도 인구수가 입력 데이터에 있으면 해당 값을 사용하고, 없으면 최근 기준 인구수를 유지합니다. 이 경우 결과 파일의 `인구수_추정_방법`에 `latest_known_population_carry_forward`가 기록됩니다.

업로드 검증 정책:

- 허용 확장자: `.csv`, `.xlsx`, `.xls`
- 기본 최대 파일 크기: 20MB
- 기본 최대 행 수: 100,000행
- Excel sheet 수: 1개만 허용
- Excel formula cell은 업로드 불가
- 파일이 아닌 경로, 존재하지 않는 경로, 허용되지 않은 확장자는 거부

## Sample Input/Output

- `data/sample_prediction_input.xlsx`: 예측 입력 예시
- `data/prediction_result_YYYY.xlsx`: 연도별 예측 결과 예시

이 파일들은 실행 중 생성되는 산출물이므로 기본적으로 Git에 포함하지 않습니다. 공개 데모가 필요하면 작은 샘플만 별도 release asset으로 제공하세요.

## 데이터 출처와 라이선스

프로젝트는 공공 범죄 통계와 인구 통계 CSV를 입력으로 사용하는 구조입니다. 공개 저장소에 데이터를 포함하려면 다음을 README 또는 `DATA.md`에 명시하세요.

- 데이터 제공 기관
- 원본 다운로드 URL
- 수집 기준일
- 라이선스 또는 이용 약관
- 전처리 과정
- 재배포 가능 여부

재배포 조건이 불명확한 원천 데이터는 저장소에 포함하지 말고, 사용자가 직접 다운로드하도록 안내하는 방식을 권장합니다.

## Pickle 보안 주의사항

`models/best_model.pkl`은 Python pickle 파일입니다. pickle은 로드 과정에서 임의 코드 실행이 가능하므로 신뢰할 수 없는 모델 파일을 절대 로드하지 마세요.

이 프로젝트는 `models/model_info.json`의 `model_sha256`과 실제 모델 파일의 SHA256을 비교한 뒤에만 모델을 로드합니다. 이 검증은 실수로 모델 파일이 바뀐 경우를 잡는 데 도움이 됩니다. 하지만 `best_model.pkl`과 `model_info.json`이 같은 디렉터리에 있고 둘 다 공격자에 의해 변조될 수 있다면 SHA256 검증만으로는 보안을 보장할 수 없습니다.

공개 배포 시 권장 정책:

- 모델 pickle 파일은 Git 저장소에 포함하지 않기
- GitHub Release asset 또는 별도 아티팩트 저장소로 배포하기
- release note에 SHA256을 별도로 게시하기
- 가능하면 서명된 체크섬 또는 신뢰된 배포 채널 사용하기
- 운영 환경에서는 pickle 대신 안전한 직렬화 포맷 또는 제한된 로더 검토하기

## 저장소 관리

Git에 포함하지 않는 항목:

- `venv/`
- `__pycache__/`
- `.pytest_cache/`
- 생성된 CSV/XLSX
- `models/best_model.pkl`
- 대용량 원천 데이터

`models/model_info.json`은 모델 메타데이터 예시로 유지할 수 있지만, 실제 배포 모델과 일치하는 최신 값인지 확인해야 합니다.

## 현재 한계

- 2025~2027 실제값은 아직 없으므로 재귀 예측은 전년도 예측 오차가 다음 연도로 누적될 수 있습니다.
- 미래 인구 데이터가 없으면 최근 인구수를 유지하는 정책을 사용하므로 장기 예측에는 인구 변동 오차가 포함될 수 있습니다.
- Random Forest와 XGBoost는 프로젝트 내부 직접 구현 모델이며, 운영 수준 최적화 라이브러리와 동일한 성능/기능을 보장하지 않습니다.
