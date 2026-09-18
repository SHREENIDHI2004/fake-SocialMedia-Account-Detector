from typing import List, Optional, Literal

try:
    from pydantic import BaseModel, Field, ConfigDict
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)
        def model_dump(self, *args, **kwargs):
            return self.__dict__.copy()
        def model_dump_json(self, *args, **kwargs):
            import json
            return json.dumps(self.__dict__, default=str)
    def Field(*args, **kwargs): return None
    class ConfigDict:
        @staticmethod
        def from_attributes(b): return {}
    class LiteralWrapper:
        def __class_getitem__(cls, item): return type(item)
    Literal = LiteralWrapper()  # type: ignore[assignment,misc]


class AccountFeatureRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    username: Optional[str] = Field(default=None, description="Account username handle")
    fullname: Optional[str] = Field(default=None, description="Display / full name on profile")
    followers: Optional[int] = Field(default=None, ge=0)
    following: Optional[int] = Field(default=None, ge=0)
    posts: Optional[int] = Field(default=None, ge=0)
    profile_pic: Optional[int] = Field(default=None, description="1=has profile picture, 0=none")
    bio_length: Optional[int] = Field(default=None, ge=0, description="Character length of bio text")
    bio_text: Optional[str] = Field(default=None)
    bio_url: Optional[int] = Field(default=None, description="1=bio contains a URL, 0=no URL")
    external_url: Optional[int] = Field(default=None, description="1=profile has external URL, 0=none")
    private: Optional[int] = Field(default=None, description="1=private, 0=public")
    verified: Optional[int] = Field(default=None, description="1=platform verified badge present, 0=not")
    account_age_days: Optional[int] = Field(default=None, ge=0, description="Days since account creation, if known")


class EvidenceItem(BaseModel):
    label: str = Field(description="Short label for this evidence point")
    detail: str = Field(description="One-sentence explanation of the observation")
    source: str = Field(description="Where this evidence came from (e.g. 'ml_feature', 'shap', 'rag')")
    magnitude: Optional[float] = Field(default=None, description="Optional numeric strength (e.g. SHAP value)")


class InvestigationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = Field(
        description="Three-tier risk band derived from the calibrated ML probability"
    )
    risk_probability: float = Field(
        ge=0.0, le=1.0, description="Calibrated ML fake-class probability"
    )
    decision_threshold: float = Field(
        ge=0.0, le=1.0, description="Classification threshold used for binary label"
    )
    is_flagged: bool = Field(
        description="True iff risk_probability >= decision_threshold"
    )
    risk_summary: str = Field(
        description="2-3 sentence grounded summary of the prediction, always citing ML evidence"
    )
    evidence: List[EvidenceItem] = Field(
        description="Structured evidence list from ML features, SHAP, and RAG"
    )
    positive_signals: List[str] = Field(
        description="Grounded factors that push the model toward GENUINE (legitimacy cues)"
    )
    missing_information: List[str] = Field(
        description="What data would materially improve confidence, if available"
    )
    recommended_actions: List[str] = Field(
        description="Actionable manual verification steps an operator can take"
    )
    model_version: str = Field(description="Model version tag that produced this prediction")
    shap_available: bool = Field(default=False)
    rag_available: bool = Field(default=False)
    llm_available: bool = Field(default=False, description="True if an LLM enhanced the narrative")
    uncertainty_note: Optional[str] = Field(
        default=None,
        description="Optional explanation of model uncertainty, near-threshold cases, or missing features",
    )
