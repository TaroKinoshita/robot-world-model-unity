"""
TODO5-7: Metadata JSON を作る

EpisodeRecorder が書き出したファイルを読んで、scene と episode(= scene × candidate)を
1 つの metadata.json で関連付ける。Unity は不要。

実行(プロジェクト直下で):
    python Assets/Python/build_metadata.py                 # Episodes/ の全 scene
    python Assets/Python/build_metadata.py --scene scene_0000

出力:
    Episodes/scene_XXXX/metadata.json                      # scene 全体(全 episode を含む)
    Episodes/scene_XXXX/candidates/cXXX/episode.json       # episode 単体(データローダ用)
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def git(project: Path, *args):
    try:
        out = subprocess.run(["git", *args], cwd=project, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def git_info(project: Path):
    commit = git(project, "rev-parse", "HEAD")
    status = git(project, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": git(project, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "note": "commit = code state when this metadata was built; data should be generated from a clean, committed tree",
    }


def package_versions(project: Path):
    lock = load(project / "Packages" / "packages-lock.json") or {}
    deps = lock.get("dependencies", {})
    keep = ["com.unity.perception", "com.unity.render-pipelines.high-definition",
            "com.unity.robotics.urdf-importer", "com.coplaydev.unity-mcp"]
    return {k: deps[k].get("version") for k in keep if k in deps}


def rel_files(scene_dir: Path, paths: dict, missing: list):
    """{name: relative path} を確認し、無いものを missing に追加して返す"""
    for rel in paths.values():
        if isinstance(rel, dict):
            rel_files(scene_dir, rel, missing)
        elif rel is not None and not (scene_dir / rel).exists():
            missing.append(rel)
    return paths


def build_episode(scene_dir: Path, scene_id: str, cand: dict, missing: list):
    cid = cand["candidate_id"]
    base = f"candidates/{cid}"
    planned = load(scene_dir / base / "planned_trajectory.json") or {}
    executed = load(scene_dir / base / "executed" / "executed_meta.json") or {}
    final = load(scene_dir / base / "final" / "final_state.json") or {}

    files = rel_files(scene_dir, {
        "planned": {
            "trajectory_json": f"{base}/planned_trajectory.json",
            "trajectory_csv": f"{base}/planned_trajectory.csv",
        },
        "actions": {
            "meta": f"{base}/actions/actions_meta.json",
            "eef_numeric": f"{base}/actions/eef_numeric.csv",
            "fullbody_numeric": f"{base}/actions/fullbody_numeric.csv",
            "eef_raster_dir": f"{base}/actions/eef_raster",
            "fullbody_raster_dir": f"{base}/actions/fullbody_raster",
        },
        "executed": {
            "meta": f"{base}/executed/executed_meta.json",
            "trajectory_csv": f"{base}/executed/executed_trajectory.csv",
            "contacts_csv": f"{base}/executed/contacts.csv",
        },
        "final": {
            "state": f"{base}/final/final_state.json",
            "robot_free_rgb": f"{base}/final/robot_free/rgb.png",
            "robot_free_depth": f"{base}/final/robot_free/depth.exr",
            "robot_free_semantic": f"{base}/final/robot_free/semantic.png",
            "target_mask": f"{base}/final/target_mask.png",
            "with_robot_rgb": f"{base}/final/with_robot/rgb.png",
        },
    }, missing)

    return {
        "episode_id": f"{scene_id}_{cid}",
        "scene_id": scene_id,
        "candidate_id": cid,
        "candidate_index": cand.get("candidate_index"),
        "spec": planned.get("spec"),
        "planned_ok": cand.get("planned_ok"),
        "num_planned_steps": planned.get("num_steps"),
        "num_executed_steps": executed.get("num_steps"),
        "tracking": executed.get("tracking"),
        "reset": executed.get("reset"),
        "labels": final.get("labels"),
        "files": files,
    }


def build_scene(project: Path, scene_dir: Path):
    scene_id = scene_dir.name
    initial = load(scene_dir / "scene_initial.json")
    if initial is None:
        print(f"[skip] {scene_id}: scene_initial.json が無い")
        return None
    camera = load(scene_dir / "camera.json") or {}
    cands = load(scene_dir / "candidates.json") or {"candidates": []}
    missing = []

    episodes = [build_episode(scene_dir, scene_id, c, missing) for c in cands["candidates"]]

    # 代表の episode から共通情報を拾う
    first = cands["candidates"][0]["candidate_id"] if cands["candidates"] else None
    planned0 = load(scene_dir / f"candidates/{first}/planned_trajectory.json") if first else {}
    executed0 = load(scene_dir / f"candidates/{first}/executed/executed_meta.json") if first else {}
    final0 = load(scene_dir / f"candidates/{first}/final/final_state.json") if first else {}
    actions0 = load(scene_dir / f"candidates/{first}/actions/actions_meta.json") if first else {}
    planned0, executed0, final0, actions0 = planned0 or {}, executed0 or {}, final0 or {}, actions0 or {}

    labels_def = load(scene_dir / "solo_annotation_definitions.json")

    meta = {
        "schema": "scene_metadata_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ids": {
            "scene_id": scene_id,
            "scene_index": initial.get("scene_index"),
            "seed": initial.get("seed"),
            "episode_ids": [e["episode_id"] for e in episodes],
            "episode_definition": "episode = one candidate trajectory executed from this scene's initial state",
        },
        "frames": {
            "world": initial.get("world_frame"),
            "camera": camera.get("camera_frame_cv"),
            "pixel": camera.get("pixel_convention"),
            "joints": "Unity ArticulationBody jointPosition (rad)",
        },
        "units": initial.get("units"),
        "time": {
            "dt": planned0.get("dt"),
            "planned_timing": "planned step i = joint target applied during physics step i, t = i * dt",
            "executed_timing": executed0.get("timing"),
            "raster_stride": actions0.get("raster_stride"),
        },
        "robot": {
            "name": planned0.get("robot", {}).get("name"),
            "joint_names": planned0.get("joint_names"),
            "tool_link": planned0.get("robot", {}).get("tool_link"),
            "joint_sign": planned0.get("robot", {}).get("joint_sign"),
            "tcp_definition": planned0.get("tcp_definition"),
            "initial_joints": initial.get("robot", {}).get("joints"),
            "drives": executed0.get("drives"),
        },
        "camera": {
            "file": "camera.json",
            "width": camera.get("width"),
            "height": camera.get("height"),
            "fov_vertical_deg": camera.get("fov_vertical_deg"),
            "K": camera.get("K"),
            "position_world": camera.get("position_world"),
            "euler_world_deg": camera.get("euler_world_deg"),
        },
        "objects": initial.get("objects"),
        "physics": initial.get("physics"),
        "planner": planned0.get("planner"),
        "criteria": {
            "goal": final0.get("goal"),
            "label_params": final0.get("label_params"),
            "target_contact": "any robot link contacted the target during execution",
            "secondary_collision": "target touched another object, or any robot link touched a non-target object",
            "target_fell": "target final height below initial height minus fall_drop_threshold_m",
            "settle": initial.get("settle"),
            "fk_validation": cands.get("fk_validation"),
        },
        "semantic_labels": labels_def if labels_def is not None else "not copied (SOLO writes it at the end of the run)",
        "versions": {
            "unity": initial.get("unity_version"),
            "packages": package_versions(project),
            "python": platform.python_version(),
            "git": git_info(project),
        },
        "initial": rel_files(scene_dir, {
            "scene_state": "scene_initial.json",
            "camera": "camera.json",
            "candidates": "candidates.json",
            "robot_free": initial.get("files", {}).get("robot_free"),
            "with_robot": initial.get("files", {}).get("with_robot"),
        }, missing),
        "episodes": episodes,
        "missing_files": sorted(set(missing)),
    }

    (scene_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    for e in episodes:
        ep = dict(e)
        ep["scene_metadata"] = "../../metadata.json"
        (scene_dir / "candidates" / e["candidate_id"] / "episode.json").write_text(
            json.dumps(ep, indent=2, ensure_ascii=False), encoding="utf-8")

    g = meta["versions"]["git"]
    print(f"[ok] {scene_id}: episodes={len(episodes)}, missing={len(meta['missing_files'])}, "
          f"git={str(g['commit'])[:8]}{' (dirty)' if g['dirty'] else ''}")
    for e in episodes:
        lab = e["labels"] or {}
        print(f"     {e['episode_id']}: goal={lab.get('goal_reached')}, secondary={lab.get('secondary_collision')}, "
              f"target_contact={lab.get('target_contact')}")
    if meta["missing_files"]:
        print("     missing:", *meta["missing_files"], sep="\n       ")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="Unity プロジェクトのルート(省略時はこのファイルの 2 つ上)")
    ap.add_argument("--episodes", default="Episodes")
    ap.add_argument("--scene", default=None, help="scene_XXXX(省略時は全部)")
    args = ap.parse_args()

    project = Path(args.project) if args.project else Path(__file__).resolve().parents[2]
    root = project / args.episodes
    scenes = [root / args.scene] if args.scene else sorted(p for p in root.glob("scene_*") if p.is_dir())
    if not scenes:
        sys.exit(f"scene が見つからない: {root}")
    ok = [build_scene(project, s) for s in scenes]
    sys.exit(0 if all(m is not None and not m["missing_files"] for m in ok) else 1)


if __name__ == "__main__":
    main()
