import sys
import os
import json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestUI:
    def test_root_page_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"<html" in r.data.lower() or b"<!doctype" in r.data.lower()

    def test_root_page_contains_form(self, client):
        r = client.get("/")
        assert b"<form" in r.data.lower()


class TestHealthEndpoint:
    def test_health_status_ok(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_model_info_present(self, client):
        r = client.get("/api/v1/health")
        data = r.get_json()
        m = data.get("model", {})
        assert "available" in m
        assert m["available"] is True

    def test_health_rag_info_present(self, client):
        r = client.get("/api/v1/health")
        data = r.get_json()
        rag = data.get("rag", {})
        assert "available" in rag

    def test_health_genai_provider_reported(self, client):
        r = client.get("/api/v1/health")
        data = r.get_json()
        genai = data.get("genai", {})
        assert "provider" in genai


class TestModelEndpoint:
    def test_model_returns_info(self, client):
        r = client.get("/api/v1/model")
        assert r.status_code == 200
        d = r.get_json()
        assert d.get("available") is True
        assert "classifier" in d
        assert "threshold" in d

    def test_model_test_metrics(self, client):
        r = client.get("/api/v1/model")
        d = r.get_json()
        tm = d.get("final_test_metrics", {})
        assert isinstance(tm, dict)
        # At least one common metric should be present
        assert any(k in tm for k in ("accuracy", "f1", "precision", "recall", "roc_auc"))


class TestDetectEndpoint:
    def test_detect_fake_account_high_risk(self, client, fake_account_payload):
        r = client.post("/api/v1/detect", json=fake_account_payload)
        assert r.status_code == 200
        d = r.get_json()
        assert d["risk_level"] in ("HIGH", "MEDIUM")
        assert d["is_flagged"] is True
        assert 0.5 <= float(d["risk_probability"]) <= 1.0

    def test_detect_legit_account_low_risk(self, client, legit_account_payload):
        r = client.post("/api/v1/detect", json=legit_account_payload)
        assert r.status_code == 200
        d = r.get_json()
        assert d["risk_level"] == "LOW"
        assert d["is_flagged"] is False
        assert 0.0 <= float(d["risk_probability"]) < 0.5

    def test_detect_minimal_input(self, client, minimal_account_payload):
        r = client.post("/api/v1/detect", json=minimal_account_payload)
        assert r.status_code == 200
        d = r.get_json()
        assert "risk_probability" in d
        assert "missing_information" in d
        assert len(d["missing_information"]) > 0

    def test_detect_returns_evidence(self, client, fake_account_payload):
        r = client.post("/api/v1/detect", json=fake_account_payload)
        d = r.get_json()
        assert "evidence" in d
        assert len(d["evidence"]) >= 1

    def test_detect_returns_positive_signals(self, client, legit_account_payload):
        r = client.post("/api/v1/detect", json=legit_account_payload)
        d = r.get_json()
        assert "positive_signals" in d
        # Legitimate account should have some positive signals
        assert len(d["positive_signals"]) >= 1

    def test_detect_recommended_actions_present(self, client, fake_account_payload):
        r = client.post("/api/v1/detect", json=fake_account_payload)
        d = r.get_json()
        assert isinstance(d.get("recommended_actions"), list)
        assert len(d["recommended_actions"]) >= 1

    def test_detect_include_features_query_param(self, client, fake_account_payload):
        r = client.post("/api/v1/detect?include_features=1", json=fake_account_payload)
        d = r.get_json()
        assert "features" in d
        assert isinstance(d["features"], dict)

    def test_detect_model_version_tagged(self, client, fake_account_payload):
        r = client.post("/api/v1/detect", json=fake_account_payload)
        d = r.get_json()
        assert "model_version" in d
        assert d["model_version"] in ("v1", "v1-unavailable")


class TestExplainEndpoint:
    def test_explain_returns_prediction_and_shap_and_rag(self, client, fake_account_payload):
        r = client.post("/api/v1/explain", json=fake_account_payload)
        assert r.status_code == 200
        d = r.get_json()
        assert "prediction" in d
        assert "shap" in d
        assert "rag" in d

    def test_explain_shap_available(self, client, fake_account_payload):
        r = client.post("/api/v1/explain", json=fake_account_payload)
        d = r.get_json()
        shap = d["shap"]
        assert shap["available"] is True
        assert len(shap.get("risk_factors", [])) >= 1

    def test_explain_rag_available(self, client, fake_account_payload):
        r = client.post("/api/v1/explain", json=fake_account_payload)
        d = r.get_json()
        rag = d["rag"]
        assert rag["available"] is True
        assert isinstance(rag.get("documents"), list)


class TestReportEndpoint:
    def test_report_returns_report_id(self, client, fake_account_payload):
        r = client.post("/api/v1/report", json=fake_account_payload)
        assert r.status_code == 200
        d = r.get_json()
        assert "report_id" in d
        assert d["report_id"].startswith("rep_")

    def test_report_contains_sections(self, client, fake_account_payload):
        r = client.post("/api/v1/report", json=fake_account_payload)
        d = r.get_json()
        assert "sections" in d
        s = d["sections"]
        assert "prediction_summary" in s
        assert "evidence_summary" in s
        assert "recommended_next_steps" in s

    def test_report_generated_at_present(self, client, fake_account_payload):
        r = client.post("/api/v1/report", json=fake_account_payload)
        d = r.get_json()
        assert "generated_at" in d
        assert isinstance(d["generated_at"], float)


class TestInvalidInput:
    def test_empty_body_400(self, client):
        r = client.post("/api/v1/detect", data=b"", content_type="application/json")
        assert r.status_code == 400
        d = r.get_json()
        assert "error" in d

    def test_invalid_json_400(self, client):
        r = client.post(
            "/api/v1/detect", data=b"not valid json{", content_type="application/json"
        )
        assert r.status_code == 400

    def test_404_handler(self, client):
        r = client.get("/api/v1/nonexistent-route")
        assert r.status_code == 404
        d = r.get_json()
        assert "error" in d

    def test_explain_empty_body_400(self, client):
        r = client.post("/api/v1/explain", data=b"", content_type="application/json")
        assert r.status_code == 400

    def test_report_empty_body_400(self, client):
        r = client.post("/api/v1/report", data=b"", content_type="application/json")
        assert r.status_code == 400


class TestGenAIFallback:
    def test_detect_without_genai_key_llm_unavailable(self, client, fake_account_payload, monkeypatch):
        monkeypatch.setenv("GENAI_PROVIDER", "none")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        # Re-import config to pick up the patched env
        import importlib
        import config as cfg_mod
        importlib.reload(cfg_mod)
        r = client.post("/api/v1/detect", json=fake_account_payload)
        d = r.get_json()
        # llm_available must be False when provider is none / no key
        assert d["llm_available"] is False
        # Risk summary should still be present (fallback narrative used)
        assert isinstance(d.get("risk_summary"), str)
        assert len(d["risk_summary"]) > 0

    def test_explain_still_works_without_llm(self, client, fake_account_payload, monkeypatch):
        monkeypatch.setenv("GENAI_PROVIDER", "none")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        r = client.post("/api/v1/explain", json=fake_account_payload)
        assert r.status_code == 200
        d = r.get_json()
        # SHAP + RAG should still be produced even without LLM
        assert d["shap"]["available"] is True


class TestRAGFallback:
    def test_rag_fallback_to_tfidf(self, monkeypatch, tmp_path):
        """Verify that the RAG module falls back to TF-IDF when sentence-transformers unavailable."""
        import importlib
        # Patch sentence_transformers import to raise
        import builtins
        orig_import = builtins.__import__

        def mocked_import(name, *args, **kwargs):
            if name == "sentence_transformers" or (
                isinstance(name, str) and name.startswith("sentence_transformers.")
            ):
                raise ImportError("sentence-transformers mocked away")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)
        import genai.rag.retrieve as rag_mod
        importlib.reload(rag_mod)
        docs = rag_mod.load_knowledge_documents()
        if not docs:
            pytest.skip("No knowledge base docs found")
        bundle = rag_mod._build_tfidf_bundle(docs)
        assert bundle.available is True
        assert bundle.backend == "tfidf+cosine"
        results = rag_mod.retrieve("follower ratio fake account", bundle.__dict__, top_k=2)
        assert isinstance(results, list)
