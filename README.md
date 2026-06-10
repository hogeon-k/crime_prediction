# Crime Prediction

범죄 발생 건수를 예측하기 위한 Python 프로젝트입니다. 정부 범죄/인구 CSV를 전처리하고, Linear Regression, Random Forest, XGBoost 계열 모델을 비교한 뒤 선택된 AI 모델로 CSV/XLSX 입력 파일 또는 Tkinter GUI에서 예측을 수행합니다.

## 주요 기능

- 범죄/인구 데이터 병합 및 전처리
- 표준 CSV/XLSX 업로드와 정부 원천 데이터 업로드 지원
- 연도 기준 Train/Test 분리: 기본값 Train 2022~2023, Test 2024
- Linear Regression, 직접 구현 Random Forest, 직접 구현 XGBoost 비교
- 전년도 발생 건수, 전년도 범죄율, 지역/범죄유형 평균 feature 생성
- 예측 결과 CSV/XLSX 저장
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

통합 GUI:

```powershell
python run_app.py
```

개별 GUI:

```powershell
python src/run_gui.py
python src/run_excel_gui.py
```

파일 예측:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from ai.predict import predict_from_file

predict_from_file("input.xlsx", "data/prediction_result.xlsx")
```

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

학습 리포트에는 기존 모델 비교에 더해 다음 실험이 함께 출력됩니다.

- RandomForest 후보군: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- XGBoost 후보군: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `reg_lambda`, `gamma`
- 각 후보 조합의 Train/Test R2, RMSE, MAE 및 Train/Test R2 gap
- 발생건수, 인구수, 범죄율 분포와 IQR 기준 이상값 수
- 지역별, 범죄유형별 발생건수 통계
- 숫자형 feature와 target 간 상관, split count 기반 importance, permutation importance
- 연도 기반 Walk-Forward 검증: 2022 -> 2023, 2022~2023 -> 2024

최종 하이퍼파라미터 선택 기준은 Test R2를 우선 최대화하고, 성능이 유사한 경우 Test RMSE, Test MAE, Train/Test R2 gap 순서로 더 작은 조합을 선택하는 방식입니다. 데이터가 2022~2024년 3개 연도뿐이므로 일반 K-Fold보다 과거 연도로 학습하고 다음 연도를 검증하는 연도 기준 검증을 사용하며, 이 결과는 최종 성능의 확정치가 아니라 일반화 가능성을 점검하는 보조 근거로 해석합니다.

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

최종 AI 모델은 Test R2를 가장 우선하고, 동률 또는 근소 차이에서는 RMSE와 MAE가 낮은 모델을 선택합니다. baseline은 별도로 리포트하지만, 프로젝트 요구상 저장 대상은 AI 모델입니다.

현재 README에 기록된 이전 실험에서는 baseline이 모든 AI 모델보다 높은 성능을 보였습니다.

| Model | Test R2 | RMSE | MAE | Unique Ratio |
|---|---:|---:|---:|---:|
| Baseline: 2023 same region/crime | 0.9548 | 1733.36 | 569.06 | - |
| Linear | 0.9513 | 1799.14 | 748.26 | 0.9009 |
| XGBoost | 0.9297 | 2160.26 | 669.84 | 0.1641 |
| Random Forest Current | 0.9411 | 1978.37 | 663.37 | 0.1424 |
| Random Forest Depth 8 | 0.9419 | 1963.84 | 679.53 | 0.6966 |
| Random Forest Depth 12 | 0.9416 | 1969.49 | 686.89 | 0.9087 |

이 결과는 전년도 동일 지역/범죄 유형 값이 매우 강한 기준선이라는 뜻입니다. 따라서 모델 성능을 주장할 때는 “AI 모델이 baseline을 이겼다”가 아니라, “baseline과 AI 후보를 비교했고, 저장 모델은 AI 후보 중 Test R2/RMSE/MAE 기준으로 선택했다”라고 설명하는 것이 정직합니다.

학습 리포트에는 R2, RMSE, MAE, SMAPE, 예측 다양성, 지역별 MAE, 범죄 유형별 MAE를 포함합니다.

## 입력 파일 형식

예측 입력 파일은 다음 컬럼을 포함해야 합니다.

- `연도`
- `지역`
- `범죄_유형`
- `인구수`

업로드 검증 정책:

- 허용 확장자: `.csv`, `.xlsx`, `.xls`
- 기본 최대 파일 크기: 20MB
- 기본 최대 행 수: 100,000행
- Excel sheet 수: 1개만 허용
- Excel formula cell은 업로드 불가
- 파일이 아닌 경로, 존재하지 않는 경로, 허용되지 않은 확장자는 거부

## Sample Input/Output

- `data/sample_prediction_input.xlsx`: 예측 입력 예시
- `data/prediction_result.xlsx`: 예측 결과 예시

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
