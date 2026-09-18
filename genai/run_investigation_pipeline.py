import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import AccountFeatureRecord
from app.services import investigate_account
from genai.rag.retrieve import load_faiss_index


def main():
    bundle = load_faiss_index()
    print(f"RAG backend: {bundle.get('backend', 'none')} (available={bundle.get('available')})")

    cases = [
        ("DEFO_FAKE_BEHAVIOR", AccountFeatureRecord(
            username="cheapfollowerbot_2847",
            fullname="",
            followers=87, following=8400, posts=0,
            profile_pic=0, bio_length=0, bio_text="", bio_url=0,
            external_url=0, private=0, verified=0,
        )),
        ("LIKELY_GENUINE", AccountFeatureRecord(
            username="sarah.travels",
            fullname="Sarah Mitchell",
            followers=12500, following=380, posts=284,
            profile_pic=1, bio_length=140, bio_text="Coffee addict, world explorer, mother of two. Sharing adventures from 34 countries ✈️", bio_url=1,
            external_url=1, private=0, verified=0,
        )),
        ("BORDERLINE_STAT_NORMAL_FAKE", AccountFeatureRecord(
            username="jason.wilson_99",
            fullname="Jason Wilson",
            followers=309, following=250, posts=34,
            profile_pic=1, bio_length=52, bio_text="Love life, love sports, always positive!", bio_url=0,
            external_url=0, private=0, verified=0,
        )),
    ]

    print()
    for tag, rec in cases:
        print("=" * 70)
        print(f"CASE: {tag}   username={rec.username}")
        print("=" * 70)
        out = investigate_account(rec, rag_bundle=bundle)
        d = out.model_dump()
        safe_d = json.loads(json.dumps(d, default=str))
        print(f"  Risk band           : {safe_d['risk_level']}")
        print(f"  Probability         : {safe_d['risk_probability']:.4f}")
        print(f"  Threshold           : {safe_d['decision_threshold']:.2f}")
        print(f"  Flagged             : {safe_d['is_flagged']}")
        print(f"  SHAP   available    : {safe_d['shap_available']}")
        print(f"  RAG    available    : {safe_d['rag_available']}")
        print(f"  LLM    available    : {safe_d['llm_available']}")
        if safe_d.get("uncertainty_note"):
            print(f"  Uncertainty         : {safe_d['uncertainty_note']}")
        print()
        print("  Summary:")
        for ln in (safe_d["risk_summary"] or "").splitlines():
            print(f"    {ln}")
        print()
        print(f"  Evidence ({len(safe_d['evidence'])}):")
        for e in safe_d["evidence"][:8]:
            src = e.get("source", "?")
            lbl = e.get("label", "")
            mag = e.get("magnitude")
            mag_str = f" mag={mag:.3f}" if isinstance(mag, (int, float)) else ""
            detail = (e.get("detail") or "")[:160].replace("\n", " ")
            print(f"    [{src:<18s}] {lbl:<40s}{mag_str}")
            if detail:
                print(f"        -> {detail}")
        if safe_d["positive_signals"]:
            print()
            print(f"  Positive signals ({len(safe_d['positive_signals'])}):")
            for p in safe_d["positive_signals"][:5]:
                print(f"    - {p}")
        print()
        print(f"  Recommended actions ({len(safe_d['recommended_actions'])}):")
        for a in safe_d["recommended_actions"][:5]:
            print(f"    - {a}")
        print(f"  Missing info ({len(safe_d['missing_information'])}):")
        for m in safe_d["missing_information"][:4]:
            print(f"    - {m}")
        print()


if __name__ == "__main__":
    main()
