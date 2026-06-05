import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.crime_service import CrimeService

crime_files = [
    "data/crime_region_2022.csv",
    "data/crime_region_2023.csv",
    "data/crime_region_2024.csv",
]

pop_files = [
    "data/pop_2022.csv",
    "data/pop_2023.csv",
    "data/pop_2024.csv",
]

service = CrimeService()


# 1. 병합 테스트
result = service.load_and_merge(crime_files, pop_files)

print("===== 병합 테스트 =====")
print(result.success)
print(result.message)

if result.data is not None:
    df = result.data
    print(df.head())
    print("shape:", df.shape)
else:
    print("데이터 없음")
    sys.exit(1)


#  2. 검증 테스트
result = service.validate(df)

print("\n===== 검증 테스트 =====")
print(result.success)
print(result.message)

if not result.success:
    sys.exit(1)

if result.data is not None:
    df = result.data


# 3. 결측치 처리 테스트
print("\n===== 결측치 처리 전 =====")
print(df.isnull().sum())

result = service.handle_missing(df)

print("\n===== 결측치 처리 테스트 =====")
print(result.success)
print(result.message)

if result.data is not None:
    df = result.data
    print("\n결측치 처리 후:")
    print(df.isnull().sum())
else:
    sys.exit(1)


#  4. 타입 변환 + 범죄율 계산 테스트
result = service.convert_types(df)

print("\n===== 타입 변환 + 범죄율 테스트 =====")
print(result.success)
print(result.message)

if result.data is not None:
    df = result.data
    print("\n컬럼 타입:")
    print(df.dtypes)
    print("\n상위 5행:")
    print(df.head().to_string(index=False))
else:
    sys.exit(1)


#  5. 범죄율 자동 검증
print("\n===== 범죄율 자동 검증 =====")

checks = {
    "범죄율 음수 없음": (df["범죄율"] >= 0).all(),
    "범죄율 NaN 없음": df["범죄율"].isna().sum() == 0,
    "범죄율 최대값 합리적": df["범죄율"].max() < 100_000,
    "발생_건수 음수 없음": (df["발생_건수"] >= 0).all(),
    "인구수 0 이하 없음": (df["인구수"] > 0).all(),
    "연도 범위 정상": df["연도"].dropna().between(2022, 2024).all(),
}

all_passed = True
for name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {status}  {name}")
    if not passed:
        all_passed = False

print()
if all_passed:
    print("모든 검증 통과")
else:
    print("일부 검증 실패 — 데이터를 확인하세요")
