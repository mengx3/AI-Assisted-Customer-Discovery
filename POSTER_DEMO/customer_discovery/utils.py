import os
from pathlib import Path
from typing import Any, Dict
import json
import joblib

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_joblib(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)

def env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)
