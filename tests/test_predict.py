import pytest
from src.predict import load_artifact, score_transaction, validate_transaction, InputValidationError

@pytest.fixture(scope="module")
def artifact():
    return load_artifact()

def valid_payload(artifact):
    return {name: 0.0 for name in artifact["feature_names"]}

def test_valid_payload_scores(artifact):
    result = score_transaction(valid_payload(artifact), artifact)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision"] in {"review", "no_review"}

def test_missing_feature_rejected(artifact):
    payload = valid_payload(artifact)
    payload.pop("Amount")
    with pytest.raises(InputValidationError, match="Missing"):
        validate_transaction(payload, artifact["feature_names"])

def test_extra_feature_rejected(artifact):
    payload = valid_payload(artifact)
    payload["unexpected"] = 1
    with pytest.raises(InputValidationError, match="Unexpected"):
        validate_transaction(payload, artifact["feature_names"])

def test_nan_rejected(artifact):
    payload = valid_payload(artifact)
    payload["Amount"] = float("nan")
    with pytest.raises(InputValidationError, match="finite"):
        validate_transaction(payload, artifact["feature_names"])
