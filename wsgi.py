import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

_APP_PY = os.path.join(ROOT, "app.py")
_spec = importlib.util.spec_from_file_location("wsgi_app_module", _APP_PY)
_app_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app_mod)

app = _app_mod.create_app()

if __name__ == "__main__":
    _app_mod.main()
