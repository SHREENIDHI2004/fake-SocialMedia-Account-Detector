import sys
from pathlib import Path

from flask import Flask

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as _cfg
from app.api import register_app


def create_app() -> Flask:
    server_cfg = _cfg.SERVER
    app = Flask(
        __name__,
        static_folder=None,
    )
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    register_app(app)
    return app


def main():
    server_cfg = _cfg.SERVER
    app = create_app()
    print(f"Starting Fake Social Media Account Detector")
    print(f"  Host: {server_cfg['host']}  Port: {server_cfg['port']}  Debug: {server_cfg['debug']}")
    print(f"  API base: /api/v1  (health, model, detect, explain)")
    print(f"  UI:     /")
    app.run(
        host=server_cfg["host"],
        port=server_cfg["port"],
        debug=bool(server_cfg["debug"]),
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
