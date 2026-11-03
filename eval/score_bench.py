#!/usr/bin/env python3
"""Score batchbench speaker turns against the labeled boundary cases.

Each labeled case marks a speaker change the app once produced at
`boundary_s`: verdict "ok" means a real change belongs there, "wrong" with no
`correct_cut_after` means the change was spurious (over-splitting). For each
engine's turns we ask:

  ok    -> is there a speaker change within ±HIT_S of boundary_s?  (recall)
  wrong -> is there NO speaker change within ±REJECT_S?             (rejection)

Cases labeled "skip", and "wrong" cases that name a different cut word, are
reported but not scored (their correct cut is a text position, not a time).

Usage:
  score_bench.py --cases cases-meeting-X.jsonl --result whisper.json fluid.json [...]
"""

import argparse
import json

HIT_S = 1.5
REJECT_S = 1.0


def speaker_changes(turns):
    """Start times (s) of turns whose speaker differs from the previous turn."""
    ordered = sorted(turns, key=lambda t: t["start_ms"])
    changes = []
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.get("speaker") is not None and cur.get("speaker") != prev.get("speaker"):
            changes.append(cur["start_ms"] / 1000.0)
    return changes


def near(changes, t, tol):
    return any(abs(c - t) <= tol for c in changes)


def score(cases, result):
    changes = speaker_changes(result["turns"])
    ok_hit = ok_n = wrong_rej = wrong_n = unscored = 0
    for c in cases:
        if c["verdict"] == "ok":
            ok_n += 1
            ok_hit += near(changes, c["boundary_s"], HIT_S)
        elif c["verdict"] == "wrong" and not c.get("correct_cut_after"):
            wrong_n += 1
            wrong_rej += not near(changes, c["boundary_s"], REJECT_S)
        else:
            unscored += 1
    return {
        "engine": result["engine"],
        "model": result.get("model"),
        "audio_min": result["audio_secs"] / 60.0,
        "timings": result["timings_secs"],
        "turns": len(result["turns"]),
        "changes": len(changes),
        "ok": (ok_hit, ok_n),
        "wrong_rejected": (wrong_rej, wrong_n),
        "unscored": unscored,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--result", nargs="+", required=True)
    args = ap.parse_args()
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    for path in args.result:
        s = score(cases, json.load(open(path, encoding="utf-8")))
        t = s["timings"]
        print(
            f'{s["engine"]:8} {s["model"] or "":26} {s["audio_min"]:5.1f} min  '
            f'transcribe {t["transcribe"]:7.1f}s  total {t["total"]:7.1f}s  '
            f'turns {s["turns"]:4}  changes {s["changes"]:4}  '
            f'ok-hit {s["ok"][0]}/{s["ok"][1]}  wrong-rejected {s["wrong_rejected"][0]}/{s["wrong_rejected"][1]}  '
            f'unscored {s["unscored"]}'
        )


if __name__ == "__main__":
    main()
