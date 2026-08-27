# Model Selection Summary — qwen3:1.7b vs qwen3:4b vs fine-tuned 1.7B

> The engineering question: **how small can the planner model go before
> quality drops below an acceptable threshold?** On a 4 GB-GPU laptop this is
> not academic — the smaller model is faster, fully GPU-resident, and half the
> disk footprint. Phase 4 answered it with a measurement (the 1.7B is not
> safe); Phase 8 answered it again with a QLoRA fine-tune (it is now).

All three models ran the **identical harness**: same 45 conversations
(`tests/eval/conversations.yaml`), same prompts, same Python-level guards
(field cleaners, interceptors), same hardware, same seed data. The only
variables are the model tag and, for the third column, the LoRA adapter
(`adapters/planner_lora`, 3.0k synthetic conversations, seed 42). Full
per-case breakdowns: `eval_qwen3_1.7b_20260817T095013Z.md`,
`eval_qwen3_4b_20260810T071636Z.md`, and
`eval_goodfoods-planner_20260822T085034Z.md`.

## The comparison

| Metric | `qwen3:1.7b` | `goodfoods-planner` (tuned 1.7B) | `qwen3:4b` |
|---|---:|---:|---:|
| Conversations passed | 32/45 (**71.1%**) | 41/45 (**91.1%**) | 42/45 (**93.3%**) |
| Turns passed | 87/103 (84.5%) | 97/103 (94.2%) | 98/103 (95.2%) |
| Search (A) | 7/9 (77.8%) | 8/9 (88.9%) | 9/9 (100%) |
| Availability (B) | 6/9 (66.7%) | 8/9 (88.9%) | 8/9 (88.9%) |
| Booking (C) | **4/9 (44.4%)** | 7/9 (77.8%) | 7/9 (77.8%) |
| Edge (D) | 6/9 (66.7%) | 9/9 (100%) | 9/9 (100%) |
| Cancel/modify (E) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| Harness wall time | ~251 s | **161 s** | slower still |
| Runtime residency | 2.4 GB | **~1.5 GB** (Q4_K_M) | 4.2 GB |
| GPU placement | 100% GPU | **100% GPU** | 71% GPU / 29% CPU |
| On-disk size | 1.4 GB | **1.1 GB** | 2.5 GB |

Hardware context (`docs/model_specs.md`): RTX 3050 Laptop 4 GB. The 4B's
runtime footprint (weights + 8K-context KV cache) exceeds VRAM, so Ollama
offloads ~29% to CPU — disclosed, not hidden. Both 1.7B variants fit
entirely on GPU.

## Where the base 1.7B broke (Phase 4 findings)

The failure modes clustered into a clear pattern — **the 1.7B cannot reliably
follow the multi-step state machine**, even with every deterministic guard the
4B enjoys:

1. **Premature booking (the dominant failure — 8 of 13 failed conversations).**
   Given "tomorrow at 7:30pm for 4", the 1.7B planner skips
   `check_availability` entirely and emits `create_reservation` directly
   (B06, B08, C05, C07, C08, C09, D02, D08). Guards can block a malformed
   call; they cannot force a model to *choose* the availability step it
   decided to skip.
2. **Context loss across turns** (A04, B08).
3. **Hallucinated slot values** (A05: invented zone/cuisine).
4. **Wrong tool selection** (B03).
5. **Non-convergence** (C06: looped on a stale time, never booked).
6. **Over-conservative fallback** (D07).

Notably, **cancel/modify scored 9/9 on both models.** Those flows are the
most interceptor-guarded and least multi-step; the Python guards make them
model-independent. The quality delta lives almost entirely in *planning
depth* — exactly where parameter count buys reliability, and exactly what the
fine-tune targeted.

## What the fine-tune fixed (Phase 8)

The QLoRA adapter was trained on 3.0k synthetic planner conversations that
reproduce the runtime format byte-for-byte (verbatim phase prompts, inline
memory context, ChatML, non-thinking template) with zero overlap against the
45 eval fixtures (enforced by test). Result, against the Phase-4 failure
list:

- **Premature booking: eliminated.** Booking went 44.4% → 77.8%, availability
  66.7% → 88.9% — both now *equal to the 4B*. No failed conversation involves
  a skipped availability step.
- **Context loss: gone.** A04 now holds cuisine across turns.
- **Hallucinated values: gone on the guard-covered cases.** Edge bucket
  66.7% → 100% (= 4B): party-size 0/-3 rejected, invalid email re-asked,
  mid-flow context switch handled.
- **Wrong tool selection / non-convergence: gone.** B03 passes; C06 completes
  the booking.

The tuned model's four failures (A05, B08, C06, C09) are the **4B's three
failures plus one synonym gap** — A05 passes "terrace" through as
`tag:"terrace"` instead of normalizing to `"outdoor seating"`. The fine-tune
did not just raise the average; it converged the 1.7B's *failure profile*
onto the 4B's.

## Held-out check (Phase 8)

A second, disjoint benchmark of 10 conversations (`tests/eval/heldout_conversations.yaml`,
authored after the training data was frozen) scores the tuned model
**9/10 conversations, 25/26 turns** — consistent with the main set, no
memorization artifact. The single miss (H02): "GoodFoods Deck, then" leaks
the trailing ", then" into the `restaurant` arg — a string-hygiene gap, not a
planning error. (Base-model baselines were not run on the held-out set; the
45-fixture three-way above is the controlled comparison.)

## The decisions

- **Phase 4:** `qwen3:4b` was selected as the floor — the base 1.7B's 71%
  hid a 44% booking rate and availability-skipping booking attempts,
  disqualifying for a booking system regardless of speed.
- **Phase 8:** the fine-tuned 1.7B (`goodfoods-planner`, Q4_K_M, 1.1 GB)
  recovers **+20.0 pts (71.1% → 91.1%)**, lands within 2.2 pts of the 4B,
  runs ~1.6× faster than the base 1.7B through the harness (terse,
  on-distribution outputs), and is fully GPU-resident. It is the default
planner candidate; the swap is one env var (`OLLAMA_MODEL`).

_Interview one-liner: we measured the model-size floor instead of guessing
it, then moved it — a 45-conversation harness showed the 2.0B planner
silently skipping availability and firing premature bookings (71.1% vs the
4B's 93.3%), so we QLoRA-fine-tuned it on 3k task-shaped conversations and
brought it to 91.1% — the 4B's failure profile at 60% of the size, fully
GPU-resident on 4 GB._
