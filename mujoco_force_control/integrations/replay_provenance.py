"""Identify the simulator definition used to produce replay states."""

import hashlib
from pathlib import Path

import mujoco


def replay_provenance() -> dict[str, str]:
    project = Path(__file__).resolve().parents[1]
    asset_dir = project / "envs" / "assets" / "panda"
    files = [project / "envs" / "peg_in_hole.py", *sorted(p for p in asset_dir.rglob("*") if p.is_file())]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(project)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"mujoco_version": mujoco.__version__, "scene_assets_sha256": digest.hexdigest()}
