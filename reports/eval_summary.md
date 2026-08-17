# Model Selection Summary — qwen3:1.7b vs qwen3:4b

> The engineering question: **how small can the planner model go before
> quality drops below an acceptable threshold?** On a 4 GB-GPU laptop this is
> not academic — the smaller model is 2.2× faster, fully GPU-resident, and
> half the disk footprint. The eval answers whether that temptation is safe.
> (It is not.)

Both models ran the **identical harness**: same 45 conversations
(`tests/eval/conversations.yaml`), same prompts, same Python-level guards
(field cleaners, interceptors), same hardware, same seed data. The only
variable is the model tag. Full per-case breakdowns:
`eval_qwen3_1.7b_20260817T095013Z.md` and `eval_qwen3_4b_20260810T071636Z.md`.

## The comparison

| Metric | `qwen3:1.7b` (2.0B params) | `qwen3:4b` (4.0B params) | Δ |
|---|---:|---:|---:|
| Conversations passed | 32/45 (**71.1%**) | 42/45 (**93.3%**) | +22.2 pts |
| Turns passed | 87/103 (84.5%) | 98/103 (95.2%) | +10.7 pts |
| Search (A) | 7/9 (77.8%) | 9/9 (100%) | +22.2 |
| Availability (B) | 6/9 (66.7%) | 8/9 (88.9%) | +22.2 |
| Booking (C) | **4/9 (44.4%)** | 7/9 (77.8%) | +33.4 |
| Edge (D) | 6/9 (66.7%) | 9/9 (100%) | +33.3 |
| Cancel/modify (E) | 9/9 (100%) | 9/9 (100%) | 0 |
| Throughput (warm) | **48.9 tok/s** | 22.3 tok/s | 1.7B is 2.2× faster |
| GPU placement | **100% GPU** | 71% GPU / 29% CPU | 1.7B wins |
| Runtime residency | 2.4 GB | 4.2 GB | 1.7B wins |
| On-disk size | 1.4 GB | 2.5 GB | 1.7B wins |

Hardware context (`docs/model_specs.md`): RTX 3050 Laptop 4 GB. The 4B's
runtime footprint (weights + 8K-context KV cache) exceeds VRAM, so Ollama
offloads ~29% to CPU — disclosed, not hidden. The 1.7B fits entirely on GPU.

## Where the 1.7B breaks

The failure modes cluster into a clear pattern — **the 1.7B cannot reliably
follow the multi-step state machine**, even with every deterministic guard the
4B enjoys:

1. **Premature booking (the dominant failure — 8 of 13 failed conversations).**
   Given "tomorrow at 7:30pm for 4", the 1.7B planner skips
   `check_availability` entirely and emits `create_reservation` directly
   (B06, B08, C05, C07, C08, C09, D02, D08). The tell is the responder's
   "email doesn't seem valid" error — the booking tool fired without the
   required field. Guards can block a malformed call; they cannot force a
   model to *choose* the availability step it decided to skip. Worst case
   C07: the premature `create_reservation` attempt fired two turns *before*
   the user declined ("actually, let me think about it") — the conversation
   produced a booking action on a flow that should never have left the
   availability phase.
2. **Context loss across turns.** A04: forgot the cuisine stated one turn
   earlier and re-asked for preferences. B08-T3: "What reservation do you
   want to make?" — lost the thread mid-flow.
3. **Hallucinated slot values.** A05: invented zone `"Bandra West"` (stored
   zone is `West`) and cuisine `"Asian"` from a "terrace dining in West"
   query; the empty result followed.
4. **Wrong tool selection.** B03: `get_seating_labels` when the user asked
   about availability.
5. **Non-convergence.** C06: looped `check_availability` with a stale time,
   never asked for email, never booked.
6. **Over-conservative fallback.** D07: asked for clarification instead of
   searching for the unknown restaurant.

Notably, **cancel/modify scored 9/9 on both models.** Those flows are the
most interceptor-guarded and least multi-step; the Python guards make them
model-independent. The quality delta lives almost entirely in *planning
depth* — exactly where parameter count buys reliability.

## Where the 4B still fails (3 conversations, triaged)

- **B08** (real, minor): Phase 2→3 transition skips the "which email?" ask.
- **C06** (fixture over-strict): agent ran `check_availability` on a fresh
  time — defensible behavior, expectation too narrow.
- **C09** (seed-data mismatch): fixture assumes a slot that is genuinely
  booked in seed data.

## The decision

**`qwen3:4b` is the floor for this workload.** The 1.7B's 71% overall hides a
44% booking rate and booking attempts that skip availability entirely —
disqualifying for a booking system regardless of speed.
The 4B costs 2× the latency and partial CPU offload, and buys state-machine
adherence. The knee in the quality/cost curve sits between 2.0B and 4.0B
parameters for multi-step tool-use planning; single-step flows tolerate the
small model fine.

_Interview one-liner: we measured the model-size floor instead of guessing it
— a 45-conversation harness, identical for both models, showed the 2.0B model
is 2.2× faster but silently skips the availability step and fires premature
booking attempts, so we selected the 4B and documented the tradeoff._
