#!/usr/bin/env python3
"""Pack a directory of KB YAML files into a ZIP module with manifest.yaml."""
import sys
import zipfile
from pathlib import Path

import yaml


def pack(yaml_dir: str, output_zip: str, manifest_data: dict) -> None:
    yaml_path = Path(yaml_dir)
    files = sorted(yaml_path.glob("*.yaml"))
    topic_files = [f.name for f in files if f.name != "manifest.yaml"]
    manifest_data["topics"] = topic_files
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.yaml", yaml.dump(manifest_data, allow_unicode=True, sort_keys=False))
        for f in files:
            zf.write(f, f.name)
    print(f"Packed {len(topic_files)} topics -> {output_zip}")


if __name__ == "__main__":
    yaml_dir = sys.argv[1]
    output = sys.argv[2]
    manifest = {
        "name": "DevOps Senior KB",
        "id": "devops-senior",
        "version": "1.0.0",
        "author": "Mockingbird Team",
        "description": "База знаний Senior DevOps: K8s, Docker, CI/CD, IaC, сети, БД, мониторинг, безопасность, облака, микросервисы",
        "specialization": "DevOps",
        "min_app_version": "0.1.0",
    }
    pack(yaml_dir, output, manifest)
