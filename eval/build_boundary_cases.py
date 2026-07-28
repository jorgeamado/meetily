#!/usr/bin/env python3
"""Audio-grounded eval cases for speaker-boundary decisions.

For every speaker transition in a meeting's saved transcript, cut a small
audio window around the boundary from the original recording, transcribe it
independently (no diarization involved), and pair it with how Meetily split
the text. The output review sheet is annotated by a human with the correct
cut; labeled cases then score the boundary-refinement LLM(s) offline.

Usage:
  build_boundary_cases.py --meeting-id <id> --audio <audio file> \
      --db "~/Library/Application Support/com.meetily.ai/meeting_minutes.sqlite" \
      --wbench <path to wbench> --whisper-model <ggml path> \
      --out <out dir> [--max-cases N] [--window 10]
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile


def tail_words(text, n=12):
    words = text.split()
    prefix = "… " if len(words) > n else ""
    return prefix + " ".join(words[-n:])


def head_words(text, n=12):
    words = text.split()
    suffix = " …" if len(words) > n else ""
    return " ".join(words[:n]) + suffix


def transcribe_window(wbench, model, audio, start, dur, tmpdir):
    wav = os.path.join(tmpdir, f"win_{start:.1f}.wav")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{max(0.0, start):.2f}", "-t", f"{dur:.2f}", "-i", audio,
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
        check=True,
    )
    out = subprocess.run(
        [wbench, model, wav, "2"], capture_output=True, text=True, check=True
    )
    lines = [l[5:].strip() for l in out.stdout.splitlines() if l.startswith("TEXT ")]
    return " ".join(l for l in lines if l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting-id", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--wbench", required=True)
    ap.add_argument("--whisper-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cases", type=int, default=25)
    ap.add_argument("--window", type=float, default=10.0)
    args = ap.parse_args()

    con = sqlite3.connect(os.path.expanduser(args.db))
    rows = con.execute(
        "SELECT COALESCE(speaker,'?'), audio_start_time, audio_end_time, transcript "
        "FROM transcripts WHERE meeting_id=? ORDER BY audio_start_time",
        (args.meeting_id,),
    ).fetchall()
    if not rows:
        sys.exit(f"no transcript rows for meeting {args.meeting_id}")

    transitions = []
    for left, right in zip(rows, rows[1:]):
        if left[0] != right[0]:
            transitions.append((left, right))
    if len(transitions) > args.max_cases:
        transitions = transitions[: args.max_cases]

    os.makedirs(args.out, exist_ok=True)
    audio = os.path.expanduser(args.audio)
    cases = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (left, right) in enumerate(transitions):
            boundary = right[1]
            start = boundary - args.window / 2
            print(f"[{i + 1}/{len(transitions)}] window at {boundary:.1f}s", file=sys.stderr)
            window_text = transcribe_window(
                args.wbench, os.path.expanduser(args.whisper_model),
                audio, start, args.window, tmpdir,
            )
            cases.append({
                "meeting_id": args.meeting_id,
                "case": i + 1,
                "boundary_s": round(boundary, 1),
                "left_speaker": left[0],
                "left_tail": tail_words(left[3]),
                "right_speaker": right[0],
                "right_head": head_words(right[3]),
                "window_transcript": window_text,
                # To be filled by the human reviewer:
                "verdict": "",          # "ok" | "wrong"
                "correct_cut_after": "" # the word the left speaker really ends on
            })

    jsonl = os.path.join(args.out, f"cases-{args.meeting_id[:16]}.jsonl")
    with open(jsonl, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    review = os.path.join(args.out, f"review-{args.meeting_id[:16]}.md")
    with open(review, "w") as f:
        f.write(f"# Boundary review — meeting {args.meeting_id}\n\n")
        f.write("For each case: does the split match what was actually said?\n")
        f.write("Fill `verdict` (ok/wrong) and, if wrong, `correct_cut_after` in the JSONL.\n\n")
        for c in cases:
            f.write(f"## Case {c['case']} — {c['boundary_s']}s\n\n")
            f.write(f"- Meetily: **{c['left_speaker']}**: \"{c['left_tail']}\" ▸ "
                    f"**{c['right_speaker']}**: \"{c['right_head']}\"\n")
            f.write(f"- Audio window says: \"{c['window_transcript']}\"\n\n")

    print(f"{len(cases)} cases -> {jsonl}\n              review sheet -> {review}")


if __name__ == "__main__":
    main()
