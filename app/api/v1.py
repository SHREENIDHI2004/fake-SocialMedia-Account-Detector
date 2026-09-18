import json
import os
import sys
import time
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, Flask, Response, jsonify, request, render_template_string
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as _cfg
from app.schemas import AccountFeatureRecord
from app.services import investigate_account
from ml.inference.predict import (
    load_model_bundle,
    predict_account,
    predict_probability,
    ModelBundle,
)
from ml.features.engineering import build_feature_frame
from ml.preprocessing.pipeline import get_feature_names_after_preprocessing

bp_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")


class RuntimeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._model_bundle: Optional[ModelBundle] = None
        self._rag_bundle: Optional[Dict[str, Any]] = None
        self._rag_load_attempted = False

    @property
    def model_bundle(self) -> Optional[ModelBundle]:
        if self._model_bundle is None:
            with self._lock:
                if self._model_bundle is None:
                    self._model_bundle = load_model_bundle()
        return self._model_bundle

    @property
    def rag_bundle(self) -> Optional[Dict[str, Any]]:
        if not self._rag_load_attempted:
            with self._lock:
                if not self._rag_load_attempted:
                    try:
                        from genai.rag.retrieve import load_faiss_index
                        self._rag_bundle = load_faiss_index()
                    except Exception as e:
                        self._rag_bundle = {"available": False, "reason": str(e)}
                    self._rag_load_attempted = True
        return self._rag_bundle


_STATE = RuntimeState()


def _rate_limit(per_minute: int = 30):
    state: Dict[str, list] = {}

    def deco(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            limit = max(1, int(per_minute or 30))
            ip = request.remote_addr or "unknown"
            now = time.time()
            history = state.get(ip, [])
            history = [t for t in history if now - t < 60]
            if len(history) >= limit:
                resp = jsonify({
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Limit is {limit}/min.",
                })
                resp.status_code = 429
                return resp
            history.append(now)
            state[ip] = history
            return fn(*args, **kwargs)
        return inner
    return deco


def _account_from_payload(payload: Dict[str, Any]) -> AccountFeatureRecord:
    rec = AccountFeatureRecord(
        username=_get_str(payload, "username") or _get_str(payload, "handle"),
        fullname=_get_str(payload, "fullname") or _get_str(payload, "display_name") or _get_str(payload, "name"),
        followers=_get_int(payload, "followers"),
        following=_get_int(payload, "following"),
        posts=_get_int(payload, "posts"),
        profile_pic=_get_int(payload, "profile_pic") or (1 if payload.get("has_profile_pic") else 0),
        bio_length=_get_int(payload, "bio_length"),
        bio_text=_get_str(payload, "bio_text") or _get_str(payload, "bio") or _get_str(payload, "description"),
        bio_url=_get_int(payload, "bio_url") or (1 if payload.get("bio_has_url") else 0),
        external_url=_get_int(payload, "external_url") or (1 if payload.get("has_external_url") else 0),
        private=_get_int(payload, "private") or (1 if payload.get("is_private") else 0),
        verified=_get_int(payload, "verified") or (1 if payload.get("is_verified") else 0),
        account_age_days=_get_int(payload, "account_age_days"),
    )
    if rec.bio_length is None and rec.bio_text:
        rec.bio_length = len(rec.bio_text)
    return rec


def _get_int(d: dict, key: str) -> Optional[int]:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v):  # NaN
            return None
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _get_str(d: dict, key: str) -> Optional[str]:
    v = d.get(key)
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    return s


@bp_v1.route("/health", methods=["GET"])
def health():
    model_bundle = _STATE.model_bundle
    rag_bundle = _STATE.rag_bundle or {"available": False}
    genai_cfg = _cfg.GENAI
    llm_ready = bool(genai_cfg.get("provider") != "none" and genai_cfg.get("api_key"))
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "model": {
            "available": bool(model_bundle and model_bundle.available),
            "reason": None if (model_bundle and model_bundle.available) else (model_bundle.reason if model_bundle else "not loaded"),
            "model_version": _cfg.DETECTION.get("model_version", "v1"),
            "threshold": model_bundle.threshold if model_bundle else _cfg.DETECTION.get("threshold"),
            "test_f1": _safe_nested(model_bundle.metadata, ["test_metrics", "f1"]) if model_bundle else None,
            "test_roc_auc": _safe_nested(model_bundle.metadata, ["test_metrics", "roc_auc"]) if model_bundle else None,
        },
        "rag": {
            "available": bool(rag_bundle.get("available")),
            "backend": rag_bundle.get("backend"),
        },
        "genai": {
            "provider": genai_cfg.get("provider"),
            "llm_ready": llm_ready,
            "model_name": genai_cfg.get("model_name"),
        },
        "api_version": "v1",
    })


@bp_v1.route("/model", methods=["GET"])
def model_info():
    model_bundle = _STATE.model_bundle
    meta = model_bundle.metadata if (model_bundle and model_bundle.available) else {}
    dataset_info = None
    if meta:
        dataset_info = {
            "train_n": meta.get("tuned_on_rows"),
            "test_n": meta.get("frozen_test_rows"),
            "model_name": meta.get("model_name"),
            "calibration_method": meta.get("calibration_method"),
        }
    return jsonify({
        "model_version": meta.get("model_version") or _cfg.DETECTION.get("model_version"),
        "available": bool(model_bundle and model_bundle.available),
        "calibrated": bool(model_bundle and model_bundle.calibrated_pipeline is not None),
        "classifier": meta.get("model_name"),
        "best_hyperparameters": meta.get("best_params"),
        "train_split_size": meta.get("tuned_on_rows"),
        "final_test_metrics": meta.get("test_metrics"),
        "dataset": dataset_info,
        "threshold": model_bundle.threshold if model_bundle else _cfg.DETECTION.get("threshold"),
    })


@bp_v1.route("/detect", methods=["POST"])
@_rate_limit(per_minute=_cfg.SERVER.get("rate_limit_per_minute", 30))
def detect():
    payload = _parse_json_payload(max_bytes=200 * 1024)
    account_payload = payload.get("account") if isinstance(payload, dict) else None
    if not isinstance(account_payload, dict):
        account_payload = payload if isinstance(payload, dict) else {}
    rec = _account_from_payload(account_payload)
    model_bundle = _STATE.model_bundle
    rag_bundle = _STATE.rag_bundle if _cfg.DETECTION.get("risk_level_medium") is not None else None

    inv = investigate_account(rec, model_bundle=model_bundle, rag_bundle=rag_bundle)
    resp_data = inv.model_dump()
    include_shap = str(request.args.get("include_shap", "0")).lower() not in ("0", "false", "no")
    include_features = str(request.args.get("include_features", "0")).lower() not in ("0", "false", "no")
    if include_features:
        resp_data["features"] = build_feature_frame([rec.model_dump()]).iloc[0].where(
            lambda s: s.notna(), None).to_dict()
    return jsonify(resp_data)


@bp_v1.route("/explain", methods=["POST"])
@_rate_limit(per_minute=_cfg.SERVER.get("rate_limit_per_minute", 30))
def explain():
    payload = _parse_json_payload(max_bytes=200 * 1024)
    account_payload = payload.get("account") if isinstance(payload, dict) else None
    if not isinstance(account_payload, dict):
        account_payload = payload if isinstance(payload, dict) else {}
    rec = _account_from_payload(account_payload)
    model_bundle = _STATE.model_bundle

    result = predict_account(rec.model_dump())
    model_bundle_obj = model_bundle if model_bundle else load_model_bundle()
    shap_data = {"available": False}
    from app.services.xai_service import shap_available, explain_account as shap_explain
    from app.services.investigation_service import _load_shap_reference
    if shap_available() and model_bundle_obj.base_pipeline is not None:
        try:
            X_df = build_feature_frame([rec.model_dump()])
            pre = model_bundle_obj.base_pipeline.named_steps.get("preprocessor")
            ref = _load_shap_reference()
            if ref is None:
                ref = X_df.copy()
            fn = get_feature_names_after_preprocessing(pre, list(ref.columns), X_ref=ref) if pre else None
            shap_data = shap_explain(model_bundle_obj.base_pipeline, ref, X_df, feature_names=fn, top_n=8)
        except Exception as e:
            shap_data = {"available": False, "error": str(e)}

    rag_bundle = _STATE.rag_bundle
    rag_docs = []
    if rag_bundle and rag_bundle.get("available"):
        from genai.rag.retrieve import retrieve
        q = f"followers={rec.followers} following={rec.following} risk {result.get('risk_probability', 0):.2f}"
        docs = retrieve(q, rag_bundle, top_k=3)
        rag_docs = [
            {"title": d["title"], "score": d["score"], "source_file": d["source_file"],
             "snippet": d["text"][:400]}
            for d in docs
        ]

    return jsonify({
        "prediction": result,
        "shap": shap_data,
        "rag": {
            "available": bool(rag_bundle and rag_bundle.get("available")),
            "backend": rag_bundle.get("backend") if rag_bundle else None,
            "documents": rag_docs,
        },
        "model_version": result.get("model_version"),
    })


@bp_v1.route("/report", methods=["POST"])
@_rate_limit(per_minute=_cfg.SERVER.get("rate_limit_per_minute", 30))
def report():
    payload = _parse_json_payload(max_bytes=200 * 1024)
    account_payload = payload.get("account") if isinstance(payload, dict) else None
    if not isinstance(account_payload, dict):
        account_payload = payload if isinstance(payload, dict) else {}
    rec = _account_from_payload(account_payload)
    model_bundle = _STATE.model_bundle
    rag_bundle = _STATE.rag_bundle
    inv = investigate_account(rec, model_bundle=model_bundle, rag_bundle=rag_bundle)
    report_data = inv.model_dump()
    include_features = str(request.args.get("include_features", "0")).lower() not in ("0", "false", "no")
    if include_features:
        report_data["features"] = build_feature_frame([rec.model_dump()]).iloc[0].where(
            lambda s: s.notna(), None).to_dict()
    report_data["generated_at"] = time.time()
    report_data["report_id"] = f"rep_{int(time.time())}_{abs(hash(rec.username or '')) % 100000:05d}"
    report_data["sections"] = {
        "prediction_summary": {
            "risk_level": inv.risk_level,
            "risk_probability": inv.risk_probability,
            "is_flagged": inv.is_flagged,
            "threshold": inv.decision_threshold,
        },
        "evidence_summary": {
            "risk_factor_count": sum(1 for e in inv.evidence if e.source != "rag"),
            "rag_document_count": sum(1 for e in inv.evidence if e.source.startswith("rag")),
            "positive_signal_count": len(inv.positive_signals),
        },
        "recommended_next_steps": inv.recommended_actions,
    }
    return jsonify(report_data)


def _parse_json_payload(max_bytes: int = 200 * 1024) -> Dict[str, Any]:
    raw = request.get_data(cache=False, as_text=False)
    if len(raw) > max_bytes:
        raise RequestEntityTooLarge(f"Request payload too large ({len(raw)} > {max_bytes})")
    if not raw:
        raise BadRequest("Empty request body")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise BadRequest(f"Invalid JSON: {e}")
    if not isinstance(payload, dict):
        raise BadRequest("Payload must be a JSON object")
    return payload


def _safe_nested(d: Optional[dict], keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


INDEX_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Fake Social Media Account Detector</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    :root { color-scheme: light dark; --bg: #0b1020; --panel: #121934; --accent: #4f8cff; --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444; --text: #e5e7eb; --muted: #9aa3b2; }
    * { box-sizing: border-box; }
    body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin:0; padding:28px; background: var(--bg); color: var(--text); }
    main { max-width: 1020px; margin: 0 auto; }
    h1 { font-size: 24px; margin: 0 0 6px; }
    .subtitle { color: var(--muted); margin-bottom: 22px; }
    .grid { display: grid; grid-template-columns: 1.1fr 1.4fr; gap: 22px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .panel { background: var(--panel); border: 1px solid #1f2a52; border-radius: 12px; padding: 20px; }
    label { display: block; margin: 10px 0 4px; font-size: 13px; color: var(--muted); }
    input, textarea, select { width:100%; padding: 9px 11px; border: 1px solid #2c396b; border-radius: 8px; background: #0e142d; color: var(--text); font-size: 14px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button { margin-top: 16px; background: var(--accent); color: white; border:0; border-radius: 8px; padding: 11px 18px; font-weight: 600; cursor: pointer; }
    button:hover { filter: brightness(1.1); }
    .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .pill.HIGH { background: var(--bad); color: #210505; }
    .pill.MEDIUM { background: var(--warn); color: #2a1800; }
    .pill.LOW { background: var(--ok); color: #03230f; }
    .pill.UNKNOWN { background: var(--muted); color: #0f1427; }
    .prob-bar { height: 10px; border-radius: 999px; background: #1c2548; overflow: hidden; margin: 10px 0 16px; }
    .prob-bar > div { height: 100%; background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444); width: 0%; transition: width .4s; }
    h3 { margin-top: 22px; font-size: 15px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
    ul { padding-left: 18px; margin: 6px 0; }
    li { margin: 4px 0; font-size: 14px; }
    .ev { border: 1px solid #26315f; border-radius: 10px; padding: 10px 12px; margin-bottom: 8px; background: #0e1532; }
    .ev .src { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
    .ev .lbl { font-weight: 600; margin-bottom: 3px; }
    .ev .det { font-size: 13px; color: #c7cee0; }
    .status-bar { font-size: 13px; color: var(--muted); margin-top: 18px; }
    code { background: #0e1532; padding: 1px 5px; border-radius: 5px; font-size: 12px; }
    details summary { cursor: pointer; color: var(--muted); font-size: 13px; margin-top: 8px; }
  </style>
</head>
<body>
<main>
  <h1>🛡️ Fake Social Media Account Detector</h1>
  <div class="subtitle">Tabular ML classifier + SHAP explainability + RAG-grounded investigation assistant</div>

  <div class="grid">
    <form class="panel" id="form" onsubmit="event.preventDefault(); runDetect();">
      <h3 style="margin-top:0">Account inputs</h3>
      <label>Username</label>
      <input name="username" placeholder="e.g. sarah.travels or bot_2847_sales" />
      <label>Display / full name</label>
      <input name="fullname" placeholder="Sarah Mitchell" />
      <div class="row">
        <div>
          <label>Followers</label>
          <input name="followers" type="number" min="0" placeholder="0" value="0" />
        </div>
        <div>
          <label>Following</label>
          <input name="following" type="number" min="0" placeholder="0" value="0" />
        </div>
      </div>
      <div class="row">
        <div>
          <label>Posts count</label>
          <input name="posts" type="number" min="0" placeholder="0" value="0" />
        </div>
        <div>
          <label>Account age (days, optional)</label>
          <input name="account_age_days" type="number" min="0" placeholder="—" />
        </div>
      </div>
      <label>Bio text</label>
      <textarea name="bio_text" rows="2" placeholder="Bio caption..."></textarea>
      <div class="row">
        <div>
          <label>Profile picture</label>
          <select name="profile_pic">
            <option value="0">No (0)</option>
            <option value="1" selected>Yes (1)</option>
          </select>
        </div>
        <div>
          <label>Verified badge</label>
          <select name="verified">
            <option value="0" selected>No (0)</option>
            <option value="1">Yes (1)</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div>
          <label>Account is private</label>
          <select name="private">
            <option value="0" selected>Public (0)</option>
            <option value="1">Private (1)</option>
          </select>
        </div>
        <div>
          <label>Has external URL</label>
          <select name="external_url">
            <option value="0" selected>No (0)</option>
            <option value="1">Yes (1)</option>
          </select>
        </div>
      </div>
      <button type="submit">🔍 Analyze account</button>
    </form>

    <section class="panel" id="result-panel">
      <h3 style="margin-top:0">Result</h3>
      <div id="placeholder">Fill in account fields and click Analyze. The API is also available at <code>/api/v1/detect</code> and <code>/api/v1/explain</code>.</div>
      <div id="result" style="display:none">
        <div style="display:flex; justify-content:space-between; align-items:center">
          <div style="font-weight:700; font-size:18px" id="r-title">—</div>
          <span class="pill" id="r-pill">UNKNOWN</span>
        </div>
        <div class="prob-bar"><div id="r-bar"></div></div>
        <div id="r-summary" style="font-size:14px; line-height:1.5"></div>
        <div id="r-uncertainty" class="status-bar" style="display:none"></div>
        <h3>Evidence</h3>
        <div id="r-evidence"></div>
        <h3>Positive legitimacy signals</h3>
        <ul id="r-positive"></ul>
        <h3>Recommended actions</h3>
        <ul id="r-actions"></ul>
        <details>
          <summary>Show missing information & pipeline status</summary>
          <ul id="r-missing"></ul>
          <div class="status-bar" id="r-status"></div>
        </details>
      </div>
    </section>
  </div>

  <div class="status-bar">Endpoints: <code>/api/v1/health</code>, <code>/api/v1/model</code>, <code>/api/v1/detect</code>, <code>/api/v1/explain</code>, <code>/api/v1/report</code></div>
</main>
<script>
async function runDetect() {
  const fd = new FormData(document.getElementById('form'));
  const data = { account: {} };
  for (const [k, v] of fd.entries()) {
    if (['followers','following','posts','account_age_days','bio_length'].includes(k)) {
      data.account[k] = v === '' ? null : parseInt(v,10);
    } else if (['profile_pic','verified','private','external_url','bio_url'].includes(k)) {
      data.account[k] = parseInt(v,10);
    } else {
      data.account[k] = v === '' ? null : v;
    }
  }
  const placeholder = document.getElementById('placeholder');
  placeholder.style.display = 'none';
  const panel = document.getElementById('result');
  panel.style.display = 'block';
  document.getElementById('r-title').textContent = 'Running model + SHAP + RAG…';
  document.getElementById('r-pill').textContent = '…';
  document.getElementById('r-bar').style.width = '0%';
  document.getElementById('r-summary').textContent = '';
  document.getElementById('r-uncertainty').style.display = 'none';
  document.getElementById('r-evidence').innerHTML = '';
  document.getElementById('r-positive').innerHTML = '';
  document.getElementById('r-actions').innerHTML = '';
  document.getElementById('r-missing').innerHTML = '';
  try {
    const resp = await fetch('/api/v1/detect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data),
    });
    if (!resp.ok) {
      let errMsg = `Server error (HTTP ${resp.status})`;
      try {
        const jerr = await resp.json();
        if (jerr.message) errMsg += ': ' + jerr.message;
        else if (jerr.error) errMsg += ': ' + jerr.error;
      } catch (_) {}
      throw new Error(errMsg);
    }
    const j = await resp.json();
    const probPct = j.risk_probability != null && !isNaN(j.risk_probability) ? (j.risk_probability * 100) : null;
    document.getElementById('r-title').textContent =
      (j.is_flagged ? '⚠️ FLAGGED — ' : '✓ Not flagged — ') +
      (probPct != null ? probPct.toFixed(1)+'% fake probability' : 'unknown');
    const pill = document.getElementById('r-pill');
    pill.textContent = j.risk_level || 'UNKNOWN';
    pill.className = 'pill ' + (j.risk_level || 'UNKNOWN');
    document.getElementById('r-bar').style.width = Math.max(0, Math.min(100, probPct != null ? probPct : 0)) + '%';
    document.getElementById('r-summary').textContent = j.risk_summary || '';
    const uncertEl = document.getElementById('r-uncertainty');
    if (j.uncertainty_note) {
      uncertEl.textContent = 'Note: ' + j.uncertainty_note;
      uncertEl.style.display = 'block';
    } else {
      uncertEl.style.display = 'none';
    }
    const ev = document.getElementById('r-evidence'); ev.innerHTML = '';
    (j.evidence || []).slice(0, 10).forEach(e => {
      const d = document.createElement('div'); d.className = 'ev';
      d.innerHTML = `<div class="src">${e.source}</div><div class="lbl">${e.label || ''}</div><div class="det">${(e.detail||'').replace(/</g,'&lt;')}</div>`;
      ev.appendChild(d);
    });
    const pos = document.getElementById('r-positive'); pos.innerHTML = '';
    (j.positive_signals || []).forEach(p => { const li = document.createElement('li'); li.textContent = p; pos.appendChild(li); });
    const acts = document.getElementById('r-actions'); acts.innerHTML = '';
    (j.recommended_actions || []).forEach(a => { const li = document.createElement('li'); li.textContent = a; acts.appendChild(li); });
    const miss = document.getElementById('r-missing'); miss.innerHTML = '';
    (j.missing_information || []).forEach(m => { const li = document.createElement('li'); li.textContent = m; miss.appendChild(li); });
    document.getElementById('r-status').textContent =
      `model: ${j.model_version}  ·  SHAP: ${j.shap_available ? 'on' : 'off'}  ·  RAG: ${j.rag_available ? 'on' : 'off'}  ·  LLM: ${j.llm_available ? 'on' : 'off'}`;
  } catch (err) {
    document.getElementById('r-title').textContent = '❌ Error';
    const pill = document.getElementById('r-pill');
    pill.textContent = 'ERROR';
    pill.className = 'pill UNKNOWN';
    document.getElementById('r-summary').textContent = 'Could not complete analysis: ' + (err.message || String(err));
    document.getElementById('r-status').textContent = 'Check server logs or try again with different input.';
  }
}
</script>
</body>
</html>
"""


def register_app(app: Flask):
    app.register_blueprint(bp_v1)

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(INDEX_PAGE)

    @app.errorhandler(400)
    def bad(e):
        return jsonify({"error": "bad_request", "message": getattr(e, "description", str(e))}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not_found"}), 404

    @app.errorhandler(413)
    def too_big(e):
        return jsonify({"error": "payload_too_large", "message": str(e)}), 413

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "internal_error", "message": "Unexpected server error"}), 500

    return app
