import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import get_config
from app.schemas import (
    AccountFeatureRecord,
    EvidenceItem,
    InvestigationResponse,
)
from ml.inference.predict import (
    ModelBundle,
    load_model_bundle,
    predict_probability,
)
from ml.features.engineering import build_feature_frame
from app.services.xai_service import explain_account, shap_available
from ml.preprocessing.pipeline import get_feature_names_after_preprocessing


_CFG = get_config()

_SHAP_REF_CACHE: Dict[str, Any] = {"df": None, "lock": None}


def _load_shap_reference() -> Optional[pd.DataFrame]:
    cached = _SHAP_REF_CACHE["df"]
    if cached is not None:
        if isinstance(cached, pd.DataFrame):
            return cached
        return None
    try:
        from pathlib import Path as _Path
        train_csv = _Path(_CFG["paths"]["kaggle_data_dir"]) / "train.csv"
        if not train_csv.exists():
            _SHAP_REF_CACHE["df"] = False  # type: ignore[assignment]
            return None
        raw = pd.read_csv(train_csv)
        sample_size = min(100, len(raw))
        if len(raw) > sample_size:
            raw = raw.sample(sample_size, random_state=42)
        records = []
        for _, r in raw.iterrows():
            pp = int(r.get("profile pic", 0))
            ext_url = int(r.get("external URL", 0))
            priv = int(r.get("private", 0))
            posts = int(r.get("#posts", 0))
            followers = int(r.get("#followers", 0))
            following = int(r.get("#follows", 0))
            bio_len = int(r.get("description length", 0))
            uname_digit_ratio = float(r.get("nums/length username", 0))
            fname_words = int(r.get("fullname words", 0))
            fname_digit_ratio = float(r.get("nums/length fullname", 0))
            name_eq_uname = int(r.get("name==username", 0))
            fake_username_digits = int(uname_digit_ratio * 12)
            fake_username = "user" + ("9" * fake_username_digits)
            fake_fullname = "A" * max(1, fname_words * 5) + (("8" * int(fname_digit_ratio * 10)) if fname_digit_ratio > 0 else "")
            records.append({
                "username": fake_username,
                "fullname": fake_fullname if not name_eq_uname else fake_username,
                "followers": followers,
                "following": following,
                "posts": posts,
                "profile_pic": pp,
                "external_url": ext_url,
                "private": priv,
                "verified": 0,
                "bio_length": bio_len,
                "bio_text": ("x" * bio_len) if bio_len > 0 else "",
            })
        ref_df = build_feature_frame(records)
        _SHAP_REF_CACHE["df"] = ref_df
        return ref_df
    except Exception:
        _SHAP_REF_CACHE["df"] = False  # type: ignore[assignment]
        return None


def _build_query_from_features(raw: AccountFeatureRecord, prob: float) -> str:
    parts = []
    parts.append(f"Risk probability {prob:.2f} (fake account classifier).")
    if raw.followers is not None or raw.following is not None:
        parts.append(f"followers={raw.followers}, following={raw.following}.")
    if raw.posts is not None:
        parts.append(f"posts={raw.posts}.")
    if raw.profile_pic in (0, 1):
        parts.append("profile picture present." if raw.profile_pic else "NO profile picture.")
    if raw.verified in (0, 1):
        parts.append("verified badge." if raw.verified else "NOT verified.")
    if raw.bio_length is not None and raw.bio_length > 0:
        parts.append(f"bio length {raw.bio_length} chars.")
    if raw.private in (0, 1):
        parts.append("private account." if raw.private else "public account.")
    return " ".join(parts)


def _build_features_input(record: AccountFeatureRecord) -> pd.DataFrame:
    row = {
        "username": record.username or "",
        "fullname": record.fullname or "",
        "followers": record.followers if record.followers is not None else 0,
        "following": record.following if record.following is not None else 0,
        "posts": record.posts if record.posts is not None else 0,
        "profile_pic": record.profile_pic if record.profile_pic is not None else 0,
        "bio_length": record.bio_length if record.bio_length is not None else 0,
        "bio_text": record.bio_text or "",
        "bio_url": record.bio_url if record.bio_url is not None else 0,
        "external_url": record.external_url if record.external_url is not None else 0,
        "private": record.private if record.private is not None else 0,
        "verified": record.verified if record.verified is not None else 0,
    }
    if record.account_age_days is not None:
        row["account_age_days"] = record.account_age_days
    return build_feature_frame([row])


def _risk_band(prob: float, cfg: Dict[str, Any]) -> str:
    if prob is None or math.isnan(prob):
        return "UNKNOWN"
    if prob >= cfg["detection"]["risk_level_high"]:
        return "HIGH"
    if prob >= cfg["detection"]["risk_level_medium"]:
        return "MEDIUM"
    return "LOW"


def _feature_evidence(raw: AccountFeatureRecord, X_df: pd.DataFrame) -> List[EvidenceItem]:
    ev: List[EvidenceItem] = []
    r = X_df.iloc[0]
    ff_ratio = float(r.get("follower_following_ratio", 0.0))
    if ff_ratio < 0.1:
        ev.append(EvidenceItem(
            label="Extreme follower/following imbalance",
            detail=f"follower/following ratio is {ff_ratio:.3f} (far below 1.0 suggests mass-following account).",
            source="ml_feature",
            magnitude=ff_ratio,
        ))
    if raw.posts is not None and int(raw.posts) == 0 and (raw.followers or 0) > 0:
        ev.append(EvidenceItem(
            label="Zero posts but nonzero followers",
            detail="The account has followers but has never posted; a common pattern for purchased shell accounts.",
            source="ml_feature",
        ))
    if raw.profile_pic == 0:
        ev.append(EvidenceItem(
            label="Missing profile picture",
            detail="No profile picture set; generic / sock-puppet marker.",
            source="ml_feature",
        ))
    if raw.verified == 0 and (raw.followers or 0) > 10000:
        ev.append(EvidenceItem(
            label="Unverified despite high follower count",
            detail="High follower count but no platform verification badge; warrants manual check.",
            source="ml_feature",
        ))
    uname = (raw.username or "")
    digits = sum(c.isdigit() for c in uname)
    if uname and digits / max(1, len(uname)) > 0.33:
        ev.append(EvidenceItem(
            label="Username with high digit ratio",
            detail=f"Username '{uname}' contains {digits} digits; auto-generated names frequently follow this pattern.",
            source="ml_feature",
        ))
    return ev


def _shap_evidence(shap_exp: Dict[str, Any]) -> Tuple[List[EvidenceItem], List[str]]:
    ev: List[EvidenceItem] = []
    positives: List[str] = []
    if not shap_exp.get("available"):
        return ev, positives
    for item in shap_exp.get("risk_factors", [])[:6]:
        ev.append(EvidenceItem(
            label=item["feature"],
            detail=f"SHAP contribution +{item['shap_value']:+.4f} toward fake class.",
            source="shap",
            magnitude=float(item["shap_value"]),
        ))
    for item in shap_exp.get("positive_signals", [])[:6]:
        positives.append(
            f"{item['feature']} (SHAP {item['shap_value']:+.4f} toward genuine class)"
        )
    return ev, positives


def _rag_evidence(rag_bundle, query: str, top_k: int) -> Tuple[List[EvidenceItem], bool]:
    if rag_bundle is None or not rag_bundle.get("available"):
        return [], False
    try:
        from genai.rag.retrieve import retrieve
    except Exception:
        return [], False
    docs = retrieve(query, rag_bundle, top_k=top_k)
    ev: List[EvidenceItem] = []
    for d in docs[:3]:
        snippet = d["text"][:240].replace("\n", " ")
        ev.append(EvidenceItem(
            label=d["title"],
            detail=f"[score={d['score']:.2f}] {snippet}",
            source=f"rag:{d['source_file']}",
            magnitude=float(d.get("score", 0.0)),
        ))
    return ev, True


def _summarize_without_llm(
    raw: AccountFeatureRecord,
    prob: float,
    threshold: float,
    band: str,
    shap_exp: Dict[str, Any],
) -> str:
    flagged = prob >= threshold
    header = (
        f"Calibrated classifier estimates a {prob:.1%} probability the account is fake "
        f"(threshold {threshold:.2f}, risk band {band}). "
    )
    if flagged:
        header += "The account is FLAGGED for potential further review. "
    else:
        header += "The account is NOT flagged by the default threshold. "
    factors = ""
    if shap_exp.get("available") and shap_exp.get("risk_factors"):
        names = [f["feature"] for f in shap_exp["risk_factors"][:3]]
        factors = f"Top ML contributors to fake score: {', '.join(names)}. "
    legit = ""
    if shap_exp.get("available") and shap_exp.get("positive_signals"):
        names = [f["feature"] for f in shap_exp["positive_signals"][:2]]
        legit = f"Countervailing legitimacy signals include: {', '.join(names)}. "
    return (header + factors + legit).strip()


def _recommended_actions(band: str, raw: AccountFeatureRecord) -> List[str]:
    actions: List[str] = []
    actions.append("Confirm follower count source and growth trajectory via platform analytics.")
    if raw.profile_pic == 0:
        actions.append("Request upload of a genuine profile picture or verified identity badge.")
    if raw.account_age_days is None:
        actions.append("Check account creation date; recent creation combined with low activity warrants caution.")
    if raw.verified == 0:
        actions.append("Encourage platform verification (or request supporting identity documents).")
    actions.append("Scan recent post captions/comments for spam keywords, copied content, or bot patterns.")
    actions.append("Review login IP history and device fingerprinting if available.")
    if band == "HIGH":
        actions.append("Escalate: place the account in a manual review queue before allowing high-value actions.")
    return actions


def _missing_info(raw: AccountFeatureRecord) -> List[str]:
    miss: List[str] = []
    if raw.account_age_days is None:
        miss.append("account creation date / account_age_days")
    if raw.bio_text is None or raw.bio_text == "":
        miss.append("bio_text content for NLP screening")
    miss.append("time-series engagement / posting frequency data")
    miss.append("device / IP authentication history")
    miss.append("follower / following network graph")
    return miss


def _try_llm_narrative(
    cfg: Dict[str, Any],
    prompt_payload: Dict[str, Any],
) -> Tuple[Optional[str], bool]:
    genai = cfg.get("genai", {})
    provider = (genai.get("provider") or "none").lower()
    key = genai.get("api_key") or ""
    if provider == "none" or not key:
        return None, False
    model_name = genai.get("model_name") or "gpt-4o-mini"
    timeout = genai.get("timeout_seconds", 15)
    import json as _json
    instructions = """
You are a SOC analyst assistant working for a social media integrity team.
You are given:
  1. A calibrated ML probability (fake account) and risk band.
  2. Evidence items derived from the account's raw fields, SHAP feature attributions,
     and RAG documents describing bot detection literature.

YOUR ROLE:
- Summarize the overall risk in 2-3 clear sentences.
- Do NOT invent any account facts beyond the provided data.
- Cite evidence IDs (e.g. "feature evidence: Missing profile picture").
- Be conservative: if information is missing, state it explicitly.
- Keep the tone professional and actionable.

Reply with exactly two sections, separated by a blank line:
RISK SUMMARY: <2-3 sentences>
UNCERTAINTY NOTE: <1 short sentence, or "None." if no missing data matters>
""".strip()
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": _json.dumps(prompt_payload, default=str, ensure_ascii=False)},
    ]
    text_out = None
    if provider == "openai":
        try:
            import urllib.request
            import ssl
            data = _json.dumps({
                "model": model_name,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 450,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
            text_out = body["choices"][0]["message"]["content"]
        except Exception:
            return None, False
    return text_out, bool(text_out)


def investigate_account(
    raw: AccountFeatureRecord,
    model_bundle: Optional[ModelBundle] = None,
    rag_bundle: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
    reference_X_df: Optional[pd.DataFrame] = None,
) -> InvestigationResponse:
    cfg = cfg or _CFG
    if model_bundle is None:
        model_bundle = load_model_bundle()

    X_df = _build_features_input(raw)
    if not model_bundle.available:
        return InvestigationResponse(
            risk_level="UNKNOWN",
            risk_probability=float("nan"),
            decision_threshold=float(cfg["detection"]["threshold"]),
            is_flagged=False,
            risk_summary="No trained model is available yet; cannot run ML prediction.",
            evidence=[],
            positive_signals=[],
            missing_information=_missing_info(raw) + ["trained model pipeline"],
            recommended_actions=[
                "Train the ML pipeline first (see ml/training/train.py)",
                "Or manually review the account using the RAG investigation guidance.",
            ],
            model_version=cfg["detection"]["model_version"],
            shap_available=False,
            rag_available=bool(rag_bundle and rag_bundle.get("available")),
            llm_available=False,
            uncertainty_note="Model bundle not available on this deployment.",
        )

    prob_raw, prob_cal = predict_probability(X_df, model_bundle)
    prob_val = float(prob_cal[0]) if prob_cal is not None else float(prob_raw[0])
    threshold = float(model_bundle.threshold or cfg["detection"]["threshold"])
    band = _risk_band(prob_val, cfg)

    evidence: List[EvidenceItem] = list(_feature_evidence(raw, X_df))

    shap_exp: Dict[str, Any] = {"available": False}
    shap_have = False
    if shap_available() and model_bundle.base_pipeline is not None:
        try:
            ref = reference_X_df
            if ref is None:
                ref = _load_shap_reference()
            if ref is None:
                ref = X_df.copy()
            pre = model_bundle.base_pipeline.named_steps.get("preprocessor")
            feature_names = get_feature_names_after_preprocessing(pre, list(ref.columns), X_ref=ref) if pre else None
            shap_exp = explain_account(
                model_bundle.base_pipeline,
                ref,
                X_df,
                feature_names=feature_names,
                top_n=5,
            )
            shap_have = bool(shap_exp.get("available"))
        except Exception:
            shap_exp = {"available": False}
    shap_ev, pos_signals = _shap_evidence(shap_exp)
    evidence.extend(shap_ev)

    query = _build_query_from_features(raw, prob_val)
    rag_ev, rag_used = _rag_evidence(rag_bundle, query, top_k=cfg["rag"]["top_k"])
    evidence.extend(rag_ev)

    base_summary = _summarize_without_llm(raw, prob_val, threshold, band, shap_exp)
    uncertainty_note = None
    llm_used = False
    if band == "MEDIUM":
        uncertainty_note = "Prediction lands in the medium band; manual verification steps are recommended."
    if abs(prob_val - threshold) < 0.10 and uncertainty_note is None:
        uncertainty_note = f"Prediction probability is within 10 points of threshold {threshold:.2f}; consider additional signals."

    llm_inputs = {
        "raw_account": raw.model_dump() if hasattr(raw, "model_dump") else raw.__dict__,
        "probability": prob_val,
        "threshold": threshold,
        "risk_band": band,
        "feature_evidence": [e.model_dump() if hasattr(e, "model_dump") else e.__dict__ for e in evidence[:18]],
        "positive_signals": pos_signals,
        "rag_doc_titles": list({e.label for e in evidence if e.source.startswith("rag")}),
    }
    llm_summary, llm_used = _try_llm_narrative(cfg, llm_inputs)
    if llm_summary:
        summary = llm_summary
        if "UNCERTAINTY NOTE:" in summary:
            parts = [p.strip() for p in summary.split("UNCERTAINTY NOTE:", 1)]
            s_main = parts[0].replace("RISK SUMMARY:", "").strip()
            s_uncert = parts[1].strip() if len(parts) > 1 else None
            if s_main:
                base_summary = s_main
            if s_uncert and s_uncert not in ("None.", "None", "none"):
                uncertainty_note = s_uncert

    missing = _missing_info(raw)
    recommended = _recommended_actions(band, raw)

    return InvestigationResponse(
        risk_level=band,  # type: ignore[arg-type]
        risk_probability=float(prob_val),
        decision_threshold=float(threshold),
        is_flagged=bool(prob_val >= threshold),
        risk_summary=base_summary,
        evidence=evidence,
        positive_signals=pos_signals,
        missing_information=missing,
        recommended_actions=recommended,
        model_version=cfg["detection"]["model_version"],
        shap_available=shap_have,
        rag_available=rag_used,
        llm_available=llm_used,
        uncertainty_note=uncertainty_note,
    )
