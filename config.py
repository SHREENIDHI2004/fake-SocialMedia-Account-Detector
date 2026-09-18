import os
from dotenv import load_dotenv

load_dotenv()


def _bool(val: str, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {'1', 'true', 'yes', 'on'}


INSTAGRAM_GRAPH_API = {
    'access_token': os.getenv('INSTAGRAM_GRAPH_API_ACCESS_TOKEN', '').strip(),
    'user_id': os.getenv('INSTAGRAM_GRAPH_API_USER_ID', '').strip(),
}

INSTAGRAM_BASIC_DISPLAY = {
    'client_id': os.getenv('INSTAGRAM_BASIC_DISPLAY_CLIENT_ID', '').strip(),
    'client_secret': os.getenv('INSTAGRAM_BASIC_DISPLAY_CLIENT_SECRET', '').strip(),
    'access_token': os.getenv('INSTAGRAM_BASIC_DISPLAY_ACCESS_TOKEN', '').strip(),
}

INSTALOADER_CONFIG = {
    'username': os.getenv('INSTALOADER_USERNAME', '').strip(),
    'password': os.getenv('INSTALOADER_PASSWORD', '').strip(),
    'use_session_file': _bool(os.getenv('INSTALOADER_USE_SESSION_FILE', 'True'), True),
}

FACEBOOK_GRAPH_API = {
    'access_token': os.getenv('FACEBOOK_GRAPH_API_ACCESS_TOKEN', '').strip(),
    'app_id': os.getenv('FACEBOOK_GRAPH_API_APP_ID', '').strip(),
    'app_secret': os.getenv('FACEBOOK_GRAPH_API_APP_SECRET', '').strip(),
}

API_METHOD = os.getenv('API_METHOD', 'mock').strip().lower()
USE_REAL_API = _bool(os.getenv('USE_REAL_API', 'False'), False)

DETECTION = {
    'model_version': os.getenv('MODEL_VERSION', 'v1').strip(),
    'threshold': float(os.getenv('DETECTION_THRESHOLD', '0.50')),
    'risk_level_high': float(os.getenv('RISK_LEVEL_HIGH', '0.70')),
    'risk_level_medium': float(os.getenv('RISK_LEVEL_MEDIUM', '0.40')),
}

GENAI = {
    'provider': os.getenv('GENAI_PROVIDER', 'none').strip().lower(),
    'api_key': os.getenv('OPENAI_API_KEY', '').strip(),
    'model_name': os.getenv('GENAI_MODEL_NAME', 'gpt-4o-mini').strip(),
    'timeout_seconds': int(os.getenv('GENAI_TIMEOUT_SECONDS', '15')),
    'max_retries': int(os.getenv('GENAI_MAX_RETRIES', '1')),
}

RAG = {
    'embedding_model': os.getenv('RAG_EMBEDDING_MODEL', 'all-MiniLM-L6-v2').strip(),
    'top_k': int(os.getenv('RAG_TOP_K', '3')),
    'index_path': os.getenv('RAG_INDEX_PATH', 'genai/rag/faiss_index.bin').strip(),
}

SERVER = {
    'host': os.getenv('FLASK_HOST', '127.0.0.1').strip(),
    'port': int(os.getenv('FLASK_PORT', '5000')),
    'debug': _bool(os.getenv('DEBUG', 'False'), False),
    'rate_limit_per_minute': int(os.getenv('RATE_LIMIT_PER_MINUTE', '30')),
}

_paths_root = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    'project_root': _paths_root,
    'data_dir': os.path.join(_paths_root, 'data'),
    'kaggle_data_dir': os.path.join(_paths_root, 'data', 'kaggle_bakhshandeh'),
    'models_dir': os.path.join(_paths_root, 'models'),
    'reports_dir': os.path.join(_paths_root, 'reports'),
    'rag_docs_dir': os.path.join(_paths_root, 'genai', 'rag', 'kb_docs'),
    'faiss_index': os.path.join(_paths_root, 'genai', 'rag', 'faiss_index.bin'),
    'threshold_path': os.path.join(_paths_root, 'models', 'v1', 'threshold.json'),
    'raw_model_path': os.path.join(_paths_root, 'models', 'v1', 'model_pipeline.joblib'),
    'calibrated_model_path': os.path.join(_paths_root, 'models', 'v1', 'calibrated_pipeline.joblib'),
    'model_metadata_path': os.path.join(_paths_root, 'models', 'v1', 'metadata.json'),
    'xai_shap_summary': os.path.join(_paths_root, 'reports', 'shap_summary_beeswarm.png'),
    'final_test_metrics_path': os.path.join(_paths_root, 'reports', 'final_test_metrics.json'),
}


def get_config() -> dict:
    return {
        'instagram_graph_api': dict(INSTAGRAM_GRAPH_API),
        'instagram_basic_display': dict(INSTAGRAM_BASIC_DISPLAY),
        'instaloader_config': dict(INSTALOADER_CONFIG),
        'facebook_graph_api': dict(FACEBOOK_GRAPH_API),
        'api_method': API_METHOD,
        'use_real_api': USE_REAL_API,
        'detection': dict(DETECTION),
        'genai': dict(GENAI),
        'rag': dict(RAG),
        'server': dict(SERVER),
        'paths': dict(PATHS),
    }
