from gui.excel_window import ExcelWindow


def test_format_prediction_error_for_missing_columns():
    message = ExcelWindow._format_prediction_error(
        "예측에 필요한 컬럼이 없습니다: ['지역']"
    )

    assert "필수 컬럼 누락" in message
    assert "지역, 범죄_유형, 연도, 인구수" in message
    assert "예측 샘플 파일" in message


def test_format_prediction_error_for_unknown_categories():
    message = ExcelWindow._format_prediction_error(
        "학습 데이터에 없는 예측 입력값이 있습니다: {'지역': ['화성']}"
    )

    assert "학습 데이터와 같은 이름" in message
    assert "예측할 수 없습니다" in message


def test_format_prediction_error_for_missing_model():
    message = ExcelWindow._format_prediction_error(
        "저장된 모델 파일이 없습니다: models/best_model.pkl"
    )

    assert "모델 파일을 찾을 수 없습니다" in message
    assert "models/best_model.pkl" in message
