import sys
import os
import warnings
import importlib.util

import pytest

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


_app_py_path = os.path.join(ROOT, "app.py")
_spec = importlib.util.spec_from_file_location("app_root_mod", _app_py_path)
_app_root_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app_root_mod)
create_app = _app_root_mod.create_app


@pytest.fixture(scope="session")
def app():
    application = create_app()
    application.config.update({"TESTING": True})
    yield application


@pytest.fixture(scope="session")
def client(app):
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def model_bundle():
    from ml.inference.predict import load_model_bundle
    return load_model_bundle()


@pytest.fixture(scope="session")
def fake_account_payload():
    return {
        "account": {
            "username": "bot_sales_9987",
            "fullname": "",
            "followers": 5,
            "following": 4900,
            "posts": 0,
            "profile_pic": 0,
            "bio_text": "win free money click link in bio giveaway crypto airdrop",
            "verified": 0,
            "private": 0,
            "external_url": 0,
            "account_age_days": 3,
        }
    }


@pytest.fixture(scope="session")
def legit_account_payload():
    return {
        "account": {
            "username": "sarah.travels",
            "fullname": "Sarah Mitchell",
            "followers": 12500,
            "following": 890,
            "posts": 342,
            "profile_pic": 1,
            "bio_text": "Adventure seeker. Sharing photos from my travels. Coffee enthusiast",
            "verified": 1,
            "private": 0,
            "external_url": 1,
            "account_age_days": 1420,
        }
    }


@pytest.fixture(scope="session")
def minimal_account_payload():
    return {
        "account": {
            "username": "test_user_123",
            "followers": 100,
            "following": 50,
        }
    }
