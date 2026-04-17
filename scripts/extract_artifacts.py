from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
MODELS_DIR = ROOT / "models"
SRC_DIR = ROOT / "src"

NOTEBOOK_PATTERN = re.compile(r"\.ipynb$", re.IGNORECASE)
PY_PATTERN = re.compile(r"\.py$", re.IGNORECASE)
MODEL_PATTERN = re.compile(r"\.(cbm|pkl|pickle|joblib)$", re.IGNORECASE)

FEATURE_VAR_NAMES = {
    "FEATURES",
    "feature_cols",
    "feature_columns",
    "selected_features",
    "final_features",
    "model_features",
    "X_cols",
    "train_cols",
    "predictors",
}

THRESHOLD_PATTERNS = [
    re.compile(r"threshold\s*=\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"best_threshold\s*=\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"optimal_threshold\s*=\s*([0-9]*\.?[0-9]+)"),
]


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="ignore")


def load_notebook_code(path: Path) -> str:
    data = json.loads(read_text_file(path))
    cells = data.get("cells", [])
    chunks: list[str] = []
    for cell in cells:
        if cell.get("cell_type") == "code":
            src = cell.get("source", [])
            if isinstance(src, list):
                chunks.append("".join(src))
            elif isinstance(src, str):
                chunks.append(src)
    return "\n\n".join(chunks)


def find_models(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and MODEL_PATTERN.search(p.name)],
        key=lambda p: str(p).lower(),
    )


def extract_catboost_features(model_path: Path) -> list[str]:
    if CatBoostClassifier is None:
        return []
    if model_path.suffix.lower() != ".cbm":
        return []
    try:
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return list(model.feature_names_) if model.feature_names_ else []
    except Exception:
        return []


def safe_literal_eval(value: str) -> Any | None:
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def extract_feature_lists_from_code(code: str) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}

    assign_pattern = re.compile(
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\[[\s\S]*?\])",
        re.MULTILINE,
    )

    for match in assign_pattern.finditer(code):
        name = match.group("name")
        if name not in FEATURE_VAR_NAMES:
            continue
        raw_value = match.group("value")
        parsed = safe_literal_eval(raw_value)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            results[name] = parsed

    return results


def extract_thresholds_from_code(code: str) -> list[float]:
    values: list[float] = []
    for pattern in THRESHOLD_PATTERNS:
        for match in pattern.finditer(code):
            try:
                values.append(float(match.group(1)))
            except Exception:
                pass
    return values


def extract_model_paths_from_code(code: str) -> list[str]:
    path_pattern = re.compile(
        r"""["']([^"']+\.(?:cbm|pkl|pickle|joblib))["']""",
        re.IGNORECASE,
    )
    return sorted(set(m.group(1) for m in path_pattern.finditer(code)))


def collect_code_sources() -> dict[str, str]:
    sources: dict[str, str] = {}

    for path in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
        sources[str(path.relative_to(ROOT))] = load_notebook_code(path)

    for path in sorted(ROOT.rglob("*.py")):
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        sources[str(path.relative_to(ROOT))] = read_text_file(path)

    return sources


def main() -> None:
    output: dict[str, Any] = {
        "repo_root": str(ROOT),
        "models_found": [],
        "catboost_feature_names": {},
        "feature_lists_from_code": {},
        "thresholds_from_code": {},
        "model_paths_from_code": {},
        "likely_best_model_candidates": [],
    }

    models = find_models(ROOT)
    output["models_found"] = [str(p.relative_to(ROOT)) for p in models]

    for model_path in models:
        features = extract_catboost_features(model_path)
        if features:
            output["catboost_feature_names"][str(model_path.relative_to(ROOT))] = features

    sources = collect_code_sources()

    for rel_path, code in sources.items():
        feature_lists = extract_feature_lists_from_code(code)
        if feature_lists:
            output["feature_lists_from_code"][rel_path] = feature_lists

        thresholds = extract_thresholds_from_code(code)
        if thresholds:
            output["thresholds_from_code"][rel_path] = thresholds

        model_paths = extract_model_paths_from_code(code)
        if model_paths:
            output["model_paths_from_code"][rel_path] = model_paths

    # Простая эвристика для кандидатов на финальную модель
    for p in output["models_found"]:
        name = Path(p).name.lower()
        score = 0
        if "tuned" in name:
            score += 3
        if "full" in name:
            score += 2
        if "catboost" in name:
            score += 1
        output["likely_best_model_candidates"].append({"path": p, "score": score})

    output["likely_best_model_candidates"] = sorted(
        output["likely_best_model_candidates"],
        key=lambda x: (-x["score"], x["path"]),
    )

    out_path = ROOT / "reports" / "artifact_scan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("НАЙДЕННЫЕ МОДЕЛИ")
    print("=" * 80)
    for p in output["models_found"]:
        print(p)

    print("\n" + "=" * 80)
    print("КАНДИДАТЫ НА ФИНАЛЬНУЮ МОДЕЛЬ")
    print("=" * 80)
    for item in output["likely_best_model_candidates"]:
        print(item)

    print("\n" + "=" * 80)
    print("FEATURE NAMES ИЗ .cbm")
    print("=" * 80)
    for model_path, features in output["catboost_feature_names"].items():
        print(f"\n{model_path}")
        print(f"Всего признаков: {len(features)}")
        print(features[:20])

    print("\n" + "=" * 80)
    print("СПИСКИ ПРИЗНАКОВ ИЗ КОДА")
    print("=" * 80)
    for rel_path, payload in output["feature_lists_from_code"].items():
        print(f"\n{rel_path}")
        for var_name, values in payload.items():
            print(f"  {var_name}: {len(values)} признаков")

    print("\n" + "=" * 80)
    print("ПОРОГИ ИЗ КОДА")
    print("=" * 80)
    for rel_path, values in output["thresholds_from_code"].items():
        print(f"{rel_path}: {values}")

    print("\n" + "=" * 80)
    print("ПУТИ К МОДЕЛЯМ ИЗ КОДА")
    print("=" * 80)
    for rel_path, values in output["model_paths_from_code"].items():
        print(f"{rel_path}: {values}")

    print(f"\nJSON сохранён: {out_path}")


if __name__ == "__main__":
    main()