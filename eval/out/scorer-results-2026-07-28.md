# Boundary tie-breaker model eval — 2026-07-28

32 human-labeled cases from 2 real meetings (12 English 3-speaker, 20 Russian
4-speaker); 22 scoreable cut decisions (5 merge-type, 3 unjudgeable, 2 beyond
candidate reach). Exact production prompts via boundary_eval + llama-helper,
greedy decoding.

| Model | Cut accuracy | Notes |
|---|---|---|
| Qwen3.5-4B Q4_K_M (production) | 17/22 (77%) | Missed 3 Russian long-shifts + 2 backchannel jiggles |
| **Qwen3.5-2B Q4_K_M** | **19/22 (86%)** | Misses subset of 4B's; ~2× faster |
| Qwen3.5-0.8B Q8_0 | 17/22 (77%) | After llama-cpp-2 upgrade to 0.1.153; ~3x faster than 2B — fine-tune candidate |

Decisions:
- **2B replaces 4B for the micro-query passes immediately** — pick_model already
  prefers it when downloaded; it is now downloaded. Faster AND more accurate.
- Fine-tuning deferred: with 2B at 86%, remaining losses are mostly NOT
  model-quality — two candidate-generation gaps (case 8: punct-trust rule
  suppresses the true +4 cut after "George."; case ru#4: +10 beyond the ±8
  window) and a keep-bias that discourages correct long jumps in Russian.
  Fixing those is cheaper than training. llama-cpp-2 upgraded to 0.1.153 same day: 0.8B now loads and scores 77%
  zero-shot — the LoRA target is closing 9 points to the 2B at 3x its speed.

## After candidate/prompt fixes (same day, later)

Fragment endings ("Sorry?") no longer suppress far candidates
(TRUSTED_SENTENCE_MIN_WORDS=3), window widened to ±12, keep-bias line now says
mid-sentence splits are usually wrong.

| Model | Before | After |
|---|---|---|
| Qwen3.5-2B | 19/24 (79%, 2 unreachable) | **20/24 (83%, 0 unreachable)** |

Remaining 4 misses share one shape: conservative "keep" on long forward jumps;
at least one (Sorry?/Sorry.) is text-ambiguous and needs acoustics — the
boundary where text-only correction tops out (see codex/web research notes:
powerset overlap posteriors + confidence-gated re-embedding are the next tier).

## Prompt ablation (2026-07-28, Qwen3.5-2B, 24 scored cases)

Theory tested: "remaining misses are anchoring bias — the model defers to the
marked current split." **Refuted.**

| variant | accuracy | notes |
|---|---|---|
| baseline (mark current + keep-bias) | **20/24 (83%)** | the 4 known long-jump misses |
| neutral (mark current, no keep instruction) | 19/24 (79%) | same misses + #4 regresses |
| blind (current split not marked) | 14/24 (58%) | 6 correct keeps break; systematic −2 drift |
| clause (baseline + long-move permission) | 19/24 (79%) | fixes nothing, breaks #9 (−4 grab) |

Conclusions:
1. The voice-analysis anchor is load-bearing: it suppresses a systematic −2
   backward drift the 2B exhibits when unanchored (+25pp over blind).
2. The 4 remaining misses are NOT prompt-fixable: even blind, the model refuses
   the long forward jumps (#18 keeps 0 with no anchor; #1 moves −6, the wrong
   direction). The answer isn't in the text — these need acoustics.
3. Prompt tuning on this axis is exhausted at 83%. Next gains come from the
   stereo channel-identity work (task #24) and the acoustic confidence layer.

Repro: `BOUNDARY_PROMPT_VARIANT={neutral|blind|clause} boundary-eval ...`

## Frontier oracle (2026-07-28, sandboxed `claude -p`, same 24 prompts)

19/24 (79%) — LOSES to the anchored local 2B (83%).
- Fixed 87b#1 (6-word forward jump) → that case IS decidable from text; 2B capability gap.
- Missed 87b#2, #18, dbea#8 — same as 2B → information-limited (answer not in text).
- Broke 87b#4 and dbea#10 which the 2B gets right.
Verdict: cloud escalation for boundary queries buys nothing (worse accuracy,
privacy cost, latency). Text ceiling ~83% confirmed by a second, much larger model.

## Acoustic probe of dbea#8 (12s window @ 89.7s, fresh diarization)

- threshold 1.1 auto: BOTH campplus and eres2net see ONE speaker in the whole window.
- forced --num-speakers 2: both split at the 95.9–96.2s pause → assigns
  "You go, George." to the NEXT speaker — contradicting George's ground truth.
- Segment boundaries identical across embedding models (segmentation-3.0 fixes
  the candidate change-points; embeddings only cluster).
Verdict: the hardest case fails text models AND embedding clustering — likely
overlapped/rapid handover. Remaining signals: channel identity (stereo, #24) or
overlap-aware powerset posteriors (research item 2).

New idea from the probe: word-gap pauses (real whisper timings) are natural
acoustic cut candidates — a future prompt could mark "this option falls on a
pause". Needs harness v2 with real word timings (current cases synthesize 300ms).

## Powerset posterior probe (2026-07-29, raw segmentation-3.0 via onnxruntime)

Hypothesis tested: overlap classes decide the information-limited misses.
REFUTED for these boundaries — overlap mass ≤0.04 at dbea#8/#9 (though real
overlap exists elsewhere: dbea case 5 shows a 2s argmax s0+1 stretch, and the
Russian 2026-07-28 call has large c6 regions).

What the probe found instead: WITHIN one 10s window the frame-level local
speaker labels carry the answer the clustering destroyed:
- dbea#8: "Sorry. You go, George." (95.2-95.9) = same local voice s2 as the
  earlier question; new voice s1 starts 96.2. Exactly George's correction.
- dbea#9: s2 ends 158.9 ("...environment."), the quiet "Yeah." blip (160.5)
  carries the NEXT speaker's label. "Yeah." belongs right.
- dbea#11: the false split sits on a 0.3s blip; first sustained new voice at
  174.8 = "I find in Turkey..." — matches ground truth via a min-duration rule.
- Caution: two windows (cases 6, 7) gave two REAL speakers the same local
  label → same-voice evidence must never veto/merge a boundary.

## Acoustic voice-change gate (2026-07-29, shipped 5830b86)

audio/segmentation.rs runs segmentation-3.0.onnx in-process (ort, already a
dependency for Parakeet; 10s window ≈ 50ms). Before each LLM boundary query:
if exactly ONE candidate cut falls inside a genuine pause between two
different sustained single-speaker voices → apply/confirm it, skip the LLM.
Guards, each earned by a real-meeting dry-run failure:
1. Overlap classes (4-6) never bound a change gap (a c6-bounded gap produced
   the only bad confirm: cut frozen mid-"потому что").
2. Gap interval must be speech-free in the RAW runs (dropped overlap/blip
   runs otherwise masquerade as pauses).
3. Short blips bound a gap only when their class is sustained elsewhere in
   the window (keeps the "Yeah." case decidable, drops isolated noise).
4. A move is suppressed when the current cut sits within 0.25s of the gap
   (inflated last-word ends: "если ‖ честно." near-miss at 1767.44).
Dry-runs (BOUNDARY_DRYRUN_FOLDER harness, committed as an ignored test):
dbea 10 boundaries → 0 fires (its refine-data has pre-DTW timestamps, up to
3.5s late — candidates cram into pauses, gate correctly stays silent);
Russian call 32 boundaries → 0 fires after guards. The gate only ever fires
when DTW word times and frame acoustics agree unambiguously; on fresh
retranscriptions expect "acoustic voice change confirms/moves" log lines.

## Gate fire-rate check (2026-07-29, live retranscribe of 57-min ru meeting)

The live run exposed a budget starvation bug: the sandwich pass produced 32
phantom-interjection queries, drained the whole 32-query budget, and pass 2
refined 0 of 32 boundaries — the acoustic gate (which costs no queries)
never ran. Fixed in 53e459a: gate is budget-exempt (runs on every boundary),
sandwich pass capped at half the budget.

Offline fire-rate on the run's FRESH DTW refine-data (dry-run harness):
0 fires / 32 boundaries. Breakdown: 23/32 have NO single-voice→pause→
single-voice gap near the cut at all (this 2-person ru call is dominated by
overlap/interruption — large c6 regions); 9/32 found a gap but no unique
candidate alignment (several current cuts sit just past the gap edge, where
the 0.25s slop guard correctly refuses precision). Verdict: hypothesis
"gate fires on fresh DTW data" REFUTED for overlap-heavy ru calls — the
gate's natural prey is clean-pause handovers (the dbea/English pattern).
It is safe everywhere but decisive only where speakers don't talk over
each other.

Suspicious observation for a future hypothesis: the 2B approved ALL 32
sandwich merges (32/32). Short replies like "Нет. Только голос.",
"согласен.", "динамику?" read as real interjections — the merge pass may
be rubber-stamping. Worth an ablation before trusting it on ru calls.
