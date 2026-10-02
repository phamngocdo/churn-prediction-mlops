from types import SimpleNamespace

import pytest

from churn_mlops.models import registry


def test_get_latest_model_uri_selects_highest_numeric_version(monkeypatch):
    captured = {}

    class FakeMlflowClient:
        def search_model_versions(self, filter_string):
            captured["filter_string"] = filter_string
            return [
                SimpleNamespace(version="9"),
                SimpleNamespace(version="12"),
                SimpleNamespace(version="3"),
            ]

    monkeypatch.setattr(registry, "MlflowClient", FakeMlflowClient)

    result = registry.get_latest_model_uri("churn_model")

    assert result == "models:/churn_model/12"
    assert captured["filter_string"] == "name='churn_model'"


def test_get_latest_model_uri_raises_when_no_versions_exist(monkeypatch):
    class FakeMlflowClient:
        def search_model_versions(self, filter_string):
            return []

    monkeypatch.setattr(registry, "MlflowClient", FakeMlflowClient)

    with pytest.raises(ValueError, match="No versions found.*churn_model"):
        registry.get_latest_model_uri("churn_model")
