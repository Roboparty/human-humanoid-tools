"""Export deterministic JSON Schema snapshots for public agent contracts."""

from __future__ import annotations

import json
from pathlib import Path

from hhtools.contracts.schema_registry import PUBLIC_AGENT_SCHEMAS


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "agent" / "v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in PUBLIC_AGENT_SCHEMAS.items():
        payload = model.model_json_schema()
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        (output_dir / f"{name}.schema.json").write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
