import hashlib
import json
import pickle

import pandas as pd
import pytest

from ai.predict import load_best_model


class PickleTestModel:
    feature_columns = ["연도", "인구수"]

    def predict(self, X):
        return pd.Series(X["인구수"], dtype=float) / 100000


def _write_model_with_info(tmp_path, expected_hash: str | None = None):
    model_path = tmp_path / "best_model.pkl"
    info_path = tmp_path / "model_info.json"

    with model_path.open("wb") as file:
        pickle.dump(PickleTestModel(), file)

    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    info_path.write_text(
        json.dumps(
            {
                "best_model": "pickle_test",
                "model_file": model_path.name,
                "model_sha256": expected_hash or model_hash,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path, info_path


def test_load_best_model_requires_matching_sha256(tmp_path):
    model_path, info_path = _write_model_with_info(tmp_path)

    model = load_best_model(model_path, info_path=info_path)

    assert isinstance(model, PickleTestModel)


def test_load_best_model_rejects_hash_mismatch(tmp_path):
    model_path, info_path = _write_model_with_info(tmp_path, expected_hash="0" * 64)

    with pytest.raises(ValueError, match="해시가 일치하지 않습니다"):
        load_best_model(model_path, info_path=info_path)


def test_load_best_model_rejects_missing_hash(tmp_path):
    model_path, info_path = _write_model_with_info(tmp_path)
    info_path.write_text(json.dumps({"best_model": "pickle_test"}), encoding="utf-8")

    with pytest.raises(ValueError, match="model_sha256"):
        load_best_model(model_path, info_path=info_path)
