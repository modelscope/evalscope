import json
from pathlib import Path
from typing import Any, Dict

PACKAGE_DIR = Path(__file__).parent
DESCRIPTION_JSON_PATH = str(PACKAGE_DIR / 'description.json')


def load_description_json(path: str = DESCRIPTION_JSON_PATH) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_description_json(data: Dict[str, Any], path: str = DESCRIPTION_JSON_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
