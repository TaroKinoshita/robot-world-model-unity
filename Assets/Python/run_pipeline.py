"""
P6: 1 シーン(Task A/B の 1 ペア)分の後処理を、決まった順に回す。
  build_metadata.py → validate_episode.py → check_task_pair.py → summarize_labels.py

BatchRunner(Unity Editor)が Play を止めたあとに別プロセスで起動する。
- 各スクリプトの出力は --log に追記する(UTF-8)
- 終わったら --result に JSON を書く(書き終わってから名前を変えるので、書きかけは読まれない)

使い方(手で回すときも同じ):
  python run_pipeline.py --episodes Episodes --scenes scene_0000 scene_0001 --pair pair_0000 \
      --log Episodes/_batch/logs/pair_0000.txt --result Episodes/_batch/logs/pair_0000.json
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def run_step(name, args, log, timeout):
    cmd = [sys.executable, str(HERE / name)] + args
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENCV_IO_ENABLE_OPENEXR="1")
    t0 = dt.datetime.now()
    log.write(f"\n## {name} {' '.join(args)}  ({t0:%Y-%m-%d %H:%M:%S})\n")
    log.flush()
    try:
        p = subprocess.run(cmd, cwd=str(PROJECT), env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        out = p.stdout.decode("utf-8", errors="replace")
        code = p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", errors="replace") + f"\n[timeout] {timeout} s\n"
        code = -9
    log.write(out)
    sec = (dt.datetime.now() - t0).total_seconds()
    log.write(f"## {name} exit {code}  ({sec:.1f} s)\n")
    log.flush()
    return {"name": name, "args": args, "exit": code, "seconds": round(sec, 1)}


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="Episodes", help="プロジェクトからの相対パス、または絶対パス")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--timeout", type=int, default=900, help="1 スクリプトあたりの上限(秒)")
    args = ap.parse_args()

    root = Path(args.episodes)
    if not root.is_absolute():
        root = PROJECT / root
    ep = ["--episodes", str(root)]

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    steps = []
    with open(args.log, "a", encoding="utf-8") as log:
        log.write(f"# run_pipeline {args.pair} {args.scenes}  ({dt.datetime.now():%Y-%m-%d %H:%M:%S})\n")
        for s in args.scenes:
            steps.append(run_step("build_metadata.py", ep + ["--scene", s], log, args.timeout))
        for s in args.scenes:
            steps.append(run_step("validate_episode.py", ep + ["--scene", s], log, args.timeout))
        steps.append(run_step("check_task_pair.py", ep + ["--pair", args.pair], log, args.timeout))
        steps.append(run_step("summarize_labels.py", ep, log, args.timeout))

        # 数字のまとめ(validation.json・pair_check.json から)
        checks = {}
        for s in args.scenes:
            v = read_json(root / s / "validation.json") or {}
            checks[s] = {"passed": v.get("passed"), "num_checks": v.get("num_checks"), "num_failed": v.get("num_failed")}
        pc = read_json(root / "pairs" / args.pair / "pair_check.json") or {}
        checks[args.pair] = {"passed": pc.get("passed"), "num_checks": pc.get("num_checks"),
                             "num_failed": pc.get("num_failed"),
                             "sim_unstable_candidates": pc.get("sim_unstable_candidates")}
        all_pass = all(st["exit"] == 0 for st in steps) and all(c.get("passed") is True for c in checks.values())
        log.write(f"# {'PASS' if all_pass else 'FAIL'} {args.pair}\n")

    out = {"pair": args.pair, "scenes": args.scenes, "pass": all_pass, "steps": steps, "checks": checks,
           "finished": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    tmp = Path(args.result + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, args.result)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
