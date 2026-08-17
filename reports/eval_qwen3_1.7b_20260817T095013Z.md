# GoodFoods eval — `qwen3:1.7b`

_Generated 20260817T095013Z (UTC)._

## Summary

- Conversations: **32/45** (71.1%)
- Turns: **87/103** (84.5%)

## By category

| Category | Pass | Total | Pass rate |
|---|---:|---:|---:|
| availability | 6 | 9 | 66.7% |
| booking | 4 | 9 | 44.4% |
| cancel_modify | 9 | 9 | 100.0% |
| edge | 6 | 9 | 66.7% |
| search | 7 | 9 | 77.8% |

## Per-conversation results

### A01 — [92mPASS[0m  (search)

_Cuisine-only triggers a clarifying reply, NOT a search. Phase-1 prompt rule (lines 82-83): cuisine + no other filter -> ask the user for additional preferences._

- [x] **T1** `I'm in the mood for Italian`

### A02 — [92mPASS[0m  (search)

_Cuisine + zone triggers a filtered search with both args._

- [x] **T1** `Any Italian places in South?`

### A03 — [92mPASS[0m  (search)

_Budget phrase normalization ("under 1000" -> max_price=1000 int) plus Asian-synonym handling ("asian food" -> cuisine "Asian")._

- [x] **T1** `Asian food under 1000`

### A04 — [91mFAIL[0m  (search)

_Multi-turn discovery. Turn 1 cuisine-only -> clarifying reply. Turn 2 adds zone -> filtered search. Validates Phase 1 holds context across turns._

- [x] **T1** `I want Italian`
- [ ] **T2** `In South`
  - reason: plan: expected plan=execute but observed plan=reply
  - reply: 'Would you like to add any other preferences such as area, budget, rating, or tags?'

### A05 — [91mFAIL[0m  (search)

_Tag extraction with synonym. Phase-1 prompt maps "terrace / patio" -> tag "outdoor seating". User says "terrace", agent emits tag="outdoor seating"._

- [ ] **T1** `terrace dining in West`
  - reason: args_subset: zone: expected 'west' observed 'Bandra West'
  - observed action: `search_restaurants_by_filters` args=`{"cuisine": "Asian", "zone": "Bandra West", "tag": "outdoor seating"}`
  - reply: 'I couldn’t find any restaurants matching those filters. Would you like to adjust the cuisine, budget, or area?'

### A06 — [92mPASS[0m  (search)

_Suggestion request with no filters -> recommend_venues with empty args._

- [x] **T1** `Any recommendations?`

### A07 — [92mPASS[0m  (search)

_User names a specific restaurant -> get_seating_labels + phase transition discovery -> availability. The most important Phase-1 transition._

- [x] **T1** `GoodFoods Grill`

### A08 — [92mPASS[0m  (search)

_Ambiguous message -> failsafe clarifying reply._

- [x] **T1** `I want food`

### A09 — [92mPASS[0m  (search)

_Ambiguous zone ("near me") + cuisine. Phase-1 synonyms section says "near me/around here" should trigger a clarifying question, NOT a search. Models vary; a search-with-zone-near-me result is a documented finding._

- [x] **T1** `Italian near me`

### B01 — [92mPASS[0m  (availability)

_Standard Phase 2 flow: restaurant + date + party_size in one turn, no time. Planner should call check_availability with time absent/null; phase stays availability (auto-transition to booking only fires when time is present)._

- [x] **T1** `GoodFoods Grill`
- [x] **T2** `18 Nov for 4 people`

### B02 — [92mPASS[0m  (availability)

_"tomorrow" date format + "table for two" party-size phrasing (=2)._

- [x] **T1** `GoodFoods Bistro`
- [x] **T2** `table for two tomorrow`

### B03 — [91mFAIL[0m  (availability)

_"we are 5" party-size phrasing + "this friday" natural date._

- [x] **T1** `GoodFoods Garden`
- [ ] **T2** `we are 5 people this friday`
  - reason: action: expected action='check_availability' observed='get_seating_labels'
  - observed action: `get_seating_labels` args=`{"restaurant": "GoodFoods Garden"}`
  - reply: 'Here are the seating options at GoodFoods Garden: Indoor dining, Corner booth, Near live kitchen, Window-side, Private dining room, Outdoor patio. Would you like me to check availability for this rest...'

### B04 — [92mPASS[0m  (availability)

_Multi-turn Phase 2: restaurant + party_size but no date -> reply asking for the date (Phase 2 collection order: date is first)._

- [x] **T1** `GoodFoods Patio`
- [x] **T2** `for 3 people`

### B05 — [92mPASS[0m  (availability)

_Multi-turn Phase 2: restaurant + date but no party_size -> reply asking for party size._

- [x] **T1** `GoodFoods Taproom`
- [x] **T2** `tomorrow`

### B06 — [91mFAIL[0m  (availability)

_Full Phase 2 with time specified -> check_availability with time, then auto-transition to booking. Ends one step from create_reservation._

- [x] **T1** `GoodFoods Grill`
- [ ] **T2** `tomorrow at 7:30pm for 4`
  - reason: action: expected action='check_availability' observed='create_reservation'
  - observed action: `create_reservation` args=`{"restaurant": "GoodFoods Grill", "date": "tomorrow", "time": "7:30pm", "party_size": 4}`
  - reply: 'The email address doesn’t seem to be valid. Would you like to try a different time?'

### B07 — [92mPASS[0m  (availability)

_User asks to see seating options mid-Phase-2. Phase-2 prompt has an explicit rule for "show/list seating" -> get_seating_labels with the memory restaurant._

- [x] **T1** `GoodFoods Grill`
- [x] **T2** `show seating options`

### B08 — [91mFAIL[0m  (availability)

_Phase 2 -> 3 transition via slot pick with email MISSING. Phase-2 prompt says reply "Which email should I use" rather than executing create_reservation._

- [x] **T1** `GoodFoods Grill`
- [ ] **T2** `tomorrow at 7:30pm for 4`
  - reason: action: expected action='check_availability' observed='create_reservation'
  - observed action: `create_reservation` args=`{"restaurant": "GoodFoods Grill", "date": "tomorrow", "time": "7:30pm", "party_size": 4}`
  - reply: 'The email address doesn’t seem to be valid. Would you like to try a different time?'
- [ ] **T3** `7:30pm works for me`
  - reason: reply_contains_any: reply matched none of ['email', 'which email']; reply='What reservation do you want to make?'
  - reply: 'What reservation do you want to make?'

### B09 — [92mPASS[0m  (availability)

_Out-of-scope input mid-Phase-2. Phase-2 prompt says IGNORE discovery / recommendations; the failsafe reply asks for date or party size._

- [x] **T1** `GoodFoods Grill`
- [x] **T2** `what's the weather like`

### C01 — [92mPASS[0m  (booking)

_Baseline happy path. "tomorrow" date, numeric party, "yes" confirm._

- [x] **T1** `GoodFoods Grill`
- [x] **T2** `tomorrow at 7:30pm for 4 people`
- [x] **T3** `john@example.com`
- [x] **T4** `yes`

### C02 — [92mPASS[0m  (booking)

_Happy path with explicit seating preference included with email._

- [x] **T1** `GoodFoods Bistro`
- [x] **T2** `tomorrow at 8pm for 2`
- [x] **T3** `jane@example.com, indoor seating`
- [x] **T4** `yes go ahead`

### C03 — [92mPASS[0m  (booking)

_"18 Nov" date format. Tests relaxed date extraction._

- [x] **T1** `GoodFoods Garden`
- [x] **T2** `18 Nov at 1:30pm for 6`
- [x] **T3** `team@example.com`
- [x] **T4** `please confirm`

### C04 — [92mPASS[0m  (booking)

_Numeric date "18/11" + "table for four" phrasing (=4)._

- [x] **T1** `GoodFoods Patio`
- [x] **T2** `18/11 at 7pm, table for four`
- [x] **T3** `pat@example.com`
- [x] **T4** `yep`

### C05 — [91mFAIL[0m  (booking)

_Larger party ("party of 8") and "next monday" natural date._

- [x] **T1** `GoodFoods Hall`
- [ ] **T2** `next monday at 7:30pm, party of 8`
  - reason: action: expected action='check_availability' observed='create_reservation'
  - observed action: `create_reservation` args=`{"restaurant": "GoodFoods Hall", "date": "next monday", "time": "7:30pm", "party_size": 8}`
  - reply: 'The email address doesn’t seem to be valid. Would you like to try a different time?'
- [x] **T3** `group@example.com`
- [x] **T4** `that's right, book it`

### C06 — [91mFAIL[0m  (booking)

_Phase 3 with one missing field: user gives restaurant+party+date but the initial time is rejected; user supplies a fresh time. Tests Phase 3 re-asking behavior without losing prior slots._

- [x] **T1** `GoodFoods Grill`
- [x] **T2** `tomorrow for 4`
- [ ] **T3** `7:30pm`
  - reason: plan: expected plan=reply but observed plan=execute
  - observed action: `check_availability` args=`{"restaurant": "GoodFoods Grill", "date": "2026-08-18", "party_size": 4, "time": "07:30"}`
  - reply: 'The available times at GoodFoods Grill on 2026-08-18 at 07:30 are: 12:00, 12:30, 13:00, 13:30, 14:00, 14:30, 15:00, 19:00, 19:30, 20:00, 20:30, 21:00, 21:30, 22:00, 22:30. Which slot works best for yo...'
- [ ] **T4** `alex@example.com`
  - reason: plan: expected plan=reply but observed plan=execute
  - observed action: `check_availability` args=`{"restaurant": "GoodFoods Grill", "date": "2026-08-18", "party_size": 4, "time": "07:30"}`
  - reply: 'The available times at GoodFoods Grill on 2026-08-18 are: 12:00, 12:30, 13:00, 13:30, 14:00, 14:30, 15:00, 19:00, 19:30, 20:00, 20:30, 21:00, 21:30, 22:00, 22:30. Which time works best for you?'
- [ ] **T5** `yes`
  - reason: action: expected action='create_reservation' observed='check_availability'
  - observed action: `check_availability` args=`{"restaurant": "GoodFoods Grill", "date": "2026-08-18", "party_size": 4, "time": "07:30"}`
  - reply: 'Great news — 12:00 is available at GoodFoods Grill on 2026-08-18. Would you like to reserve this time?'

- outcome **booking_created**: expected booking_created but create_reservation never fired

### C07 — [91mFAIL[0m  (booking)

_Negative confirmation: user declines to book. No create_reservation call, no DB row. Tests that the agent doesn't book on a "no"._

- [x] **T1** `GoodFoods Grill`
- [ ] **T2** `tomorrow at 7:30pm for 4`
  - reason: action: expected action='check_availability' observed='create_reservation'
  - observed action: `create_reservation` args=`{"restaurant": "GoodFoods Grill", "date": "tomorrow", "time": "7:30pm", "party_size": 4}`
  - reply: 'The email address doesn’t seem to be valid. Would you like to try a different time?'
- [x] **T3** `sam@example.com`
- [x] **T4** `actually, let me think about it`

- outcome **no_action**: expected no_action but fired ['create_reservation']

### C08 — [91mFAIL[0m  (booking)

_Confirmation phrase variation "sounds good" still triggers booking._

- [x] **T1** `GoodFoods Trattoria`
- [ ] **T2** `tomorrow at 7pm for 3`
  - reason: action: expected action='check_availability' observed='create_reservation'
  - observed action: `create_reservation` args=`{"restaurant": "GoodFoods Trattoria", "date": "tomorrow", "time": "19:00", "party_size": 3}`
  - reply: 'The email address doesn’t seem to be valid. Would you like to try a different time?'
- [x] **T3** `marco@example.com`
- [x] **T4** `sounds good`

### C09 — [91mFAIL[0m  (booking)

_Seating preference explicitly mentioned in the final confirmation turn. Tests that seating_pref flows into create_reservation args._

- [x] **T1** `GoodFoods Rooftop`
- [ ] **T2** `tomorrow at 6:30pm for 2`
  - reason: action: expected action='check_availability' observed='create_reservation'
  - observed action: `create_reservation` args=`{"restaurant": "GoodFoods Rooftop", "date": "tomorrow", "time": "18:30", "party_size": 2}`
  - reply: 'The email address doesn’t seem to be valid. Would you like to try a different time?'
- [x] **T3** `sky@example.com, outdoor seating please`
- [x] **T4** `yes`

### D01 — [92mPASS[0m  (edge)

_Unavailable time slot. User requests a specific time; check_availability should return is_available=false with alternatives. Planner should not proceed to booking with an unavailable slot._

- [x] **T1** `GoodFoods Grill`
- [x] **T2** `tomorrow at 3:00am for 4`

### D02 — [91mFAIL[0m  (edge)

_Invalid email format. User provides a non-email string when asked for an email. The safe-extraction path should reject it and re-ask._

- [x] **T1** `GoodFoods Grill`
- [ ] **T2** `tomorrow at 7:30pm for 4`
  - reason: action: expected action='check_availability' observed='create_reservation'
  - observed action: `create_reservation` args=`{"restaurant": "GoodFoods Grill", "date": "tomorrow", "time": "7:30pm", "party_size": 4}`
  - reply: 'The email address doesn’t seem to be valid. Would you like to try a different time?'
- [x] **T3** `my email is notanemail`

### D03 — [92mPASS[0m  (edge)

_Hallucination guard: party_size=0. safe_extract_party_size rejects 0 (valid range 1-50). Planner should not store 0; should re-ask._

- [x] **T1** `GoodFoods Grill`
- [x] **T2** `tomorrow for 0 people`

### D04 — [92mPASS[0m  (edge)

_Hallucination guard: negative party size. Should be rejected just like 0._

- [x] **T1** `GoodFoods Grill`
- [x] **T2** `tomorrow for -3 people`

### D05 — [92mPASS[0m  (edge)

_Ambiguous restaurant name. "GoodFoods" matches all 50 venues. The fuzzy resolver picks the first ilike match; behavior is non-deterministic but should NOT crash. Documented finding either way._

- [x] **T1** `GoodFoods`

### D06 — [92mPASS[0m  (edge)

_Out-of-scope question at Phase 1. Failsafe reply, no tool call._

- [x] **T1** `what's the weather today`

### D07 — [91mFAIL[0m  (edge)

_Restaurant not in DB. Search returns empty; agent should not crash and should communicate the empty result or suggest alternatives._

- [ ] **T1** `FakeFoods Place`
  - reason: plan: expected plan=execute but observed plan=reply
  - reply: "Could you clarify what you'd like to eat or where you'd like to dine?"

### D08 — [91mFAIL[0m  (edge)

_Mid-conversation context switch: user changes party size after already providing it. Memory should update to the new value, not retain the old._

- [x] **T1** `GoodFoods Grill`
- [ ] **T2** `tomorrow at 7:30pm for 4`
  - reason: action: expected action='check_availability' observed='create_reservation'
  - observed action: `create_reservation` args=`{"restaurant": "GoodFoods Grill", "date": "tomorrow", "time": "7:30pm", "party_size": 4}`
  - reply: 'The email address doesn’t seem to be valid. Would you like to try a different time?'
- [x] **T3** `actually make it 6 people`

### D09 — [92mPASS[0m  (edge)

_Ambiguous time: user offers two times in one message. Behavior is non-deterministic; planner should either ask for clarification or pick one. Documented finding._

- [x] **T1** `GoodFoods Grill`
- [x] **T2** `tomorrow at 7 or 8pm for 4`

### E01 — [92mPASS[0m  (cancel_modify)

_Cancel happy path. Seeded reservation gets soft-deleted._

- [x] **T1** `cancel my booking`
- [x] **T2** `cancel@example.com`
- [x] **T3** `yes cancel it`

### E02 — [92mPASS[0m  (cancel_modify)

_Modify time happy path. Seeded reservation's time changes; modify_reservation re-validates the new slot via get_available_slots._

- [x] **T1** `modify my reservation`
- [x] **T2** `modify@example.com`
- [x] **T3** `change the time to 20:00`

### E03 — [92mPASS[0m  (cancel_modify)

_Cancel with email already supplied in turn 1 (no clarifying reply)._

- [x] **T1** `cancel booking for inline@example.com`
- [x] **T2** `yes please cancel`

### E04 — [92mPASS[0m  (cancel_modify)

_Modify the date (not just the time). Re-validates availability._

- [x] **T1** `modify my reservation for date@example.com`
- [x] **T2** `change the date to 2025-12-05`

### E05 — [92mPASS[0m  (cancel_modify)

_Modify party size. Re-validates capacity for the new size._

- [x] **T1** `change my booking for party@example.com`
- [x] **T2** `make it 6 people instead`

### E06 — [92mPASS[0m  (cancel_modify)

_Cancel with NO active reservation. cancel_reservation returns {ok: False, error: "No active reservation for ..."}; agent should surface a friendly message, not crash._

- [x] **T1** `cancel my booking for ghost@example.com`
- [x] **T2** `yes cancel it`

### E07 — [92mPASS[0m  (cancel_modify)

_Modify with NO active reservation. Same pattern as E06 for modify_reservation._

- [x] **T1** `modify booking for nobody@example.com`
- [x] **T2** `change to 8pm`

### E08 — [92mPASS[0m  (cancel_modify)

_Modify to an unavailable slot. modify_reservation returns {ok: False, error: "...not available...", available_slots: [...]} and DOES NOT change the reservation. Seeded row should remain unchanged._

- [x] **T1** `modify my booking for unavail@example.com`
- [x] **T2** `change the time to 3:00am`

### E09 — [92mPASS[0m  (cancel_modify)

_User checks booking then changes mind ("actually don't change anything"). Should not call cancel_ or modify_reservation. Seeded row remains intact._

- [x] **T1** `check my booking for look@example.com`
- [x] **T2** `actually leave it as is`
