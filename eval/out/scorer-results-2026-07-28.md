# Boundary tie-breaker model eval — 2026-07-28

32 human-labeled cases from 2 real meetings (12 English 3-speaker, 20 Russian
4-speaker); 22 scoreable cut decisions (5 merge-type, 3 unjudgeable, 2 beyond
candidate reach). Exact production prompts via boundary_eval + llama-helper,
greedy decoding.

| Model | Cut accuracy | Notes |
|---|---|---|
| Qwen3.5-4B Q4_K_M (production) | 17/22 (77%) | Missed 3 Russian long-shifts + 2 backchannel jiggles |
| **Qwen3.5-2B Q4_K_M** | **19/22 (86%)** | Misses subset of 4B's; ~2× faster |
| Qwen3.5-0.8B Q8_0 | n/a | GGUF v3 valid but llama-cpp-2 0.1.146 cannot load it |

Decisions:
- **2B replaces 4B for the micro-query passes immediately** — pick_model already
  prefers it when downloaded; it is now downloaded. Faster AND more accurate.
- Fine-tuning deferred: with 2B at 86%, remaining losses are mostly NOT
  model-quality — two candidate-generation gaps (case 8: punct-trust rule
  suppresses the true +4 cut after "George."; case ru#4: +10 beyond the ±8
  window) and a keep-bias that discourages correct long jumps in Russian.
  Fixing those is cheaper than training. Revisit 0.8B LoRA (pop-os RTX 3080)
  after a llama-cpp-2 upgrade, if sub-second-per-boundary ever matters.
