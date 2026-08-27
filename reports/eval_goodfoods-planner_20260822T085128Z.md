# GoodFoods eval — `goodfoods-planner`

_Generated 20260822T085128Z (UTC)._

## Summary

- Conversations: **9/10** (90.0%)
- Turns: **25/26** (96.2%)

## By category

| Category | Pass | Total | Pass rate |
|---|---:|---:|---:|
| availability | 3 | 3 | 100.0% |
| booking | 3 | 3 | 100.0% |
| edge | 2 | 2 | 100.0% |
| search | 1 | 2 | 50.0% |

## Per-conversation results

### H01 — [92mPASS[0m  (search)

_Cuisine plus a zone in a single turn. Two filters are present, so the planner must search immediately rather than asking for more preferences._

- [x] **T1** `We fancy Mediterranean, ideally North-West`

### H02 — [91mFAIL[0m  (search)

_Budget-constrained discovery, then the user picks one of the results by name — which must hand off to Phase 2 via get_seating_labels._

- [x] **T1** `Anything in Central that stays under 900 a head?`
- [ ] **T2** `GoodFoods Deck, then`
  - reason: args_subset: restaurant: expected 'GoodFoods Deck' observed 'GoodFoods Deck, then'
  - observed action: `get_seating_labels` args=`{"restaurant": "GoodFoods Deck, then"}`
  - reply: 'I couldn’t find any restaurants matching those filters. Would you like to try again?'

### H03 — [92mPASS[0m  (availability)

_Date and party size arrive together with no time. The planner should run an open availability check rather than inventing a time._

- [x] **T1** `GoodFoods Brasserie`
- [x] **T2** `We'll be 6 people on 14 Dec`

### H04 — [92mPASS[0m  (availability)

_A specific dinner-service time is offered up front. The slot must be verified with check_availability before the conversation may reach booking — this is the exact step the 1.7B skips._

- [x] **T1** `GoodFoods Chophouse`
- [x] **T2** `Table for 4 on 3 Feb at 8pm`

### H05 — [92mPASS[0m  (availability)

_Party size first, date missing. Collection order is date then party size, so the planner must ask for the date and only check once it has both._

- [x] **T1** `GoodFoods Alcove`
- [x] **T2** `There will be 8 guests`
- [x] **T3** `Next saturday, please`

### H06 — [92mPASS[0m  (booking)

_Full happy path: select, verify the slot, hand over an email, confirm. create_reservation may only fire on the final turn._

- [x] **T1** `GoodFoods Vault`
- [x] **T2** `6 guests on 11 Jan at 7:30pm`
- [x] **T3** `ines.torres@example.com`
- [x] **T4** `Yes, please confirm`

### H07 — [92mPASS[0m  (booking)

_Seating preference supplied alongside the email; it must survive into memory and reach create_reservation's args._

- [x] **T1** `GoodFoods Terrace`
- [x] **T2** `4 people on 19 Jan, 8:30pm`
- [x] **T3** `wes.oduya@example.org, and we'd like to sit outside`
- [x] **T4** `Perfect, go ahead`

### H08 — [92mPASS[0m  (booking)

_The user backs out at the confirmation step. Everything needed for a booking is in memory, which is precisely when a weak planner fires create_reservation anyway — it must not._

- [x] **T1** `GoodFoods Cellar`
- [x] **T2** `2 people on 27 Jan at 9pm`
- [x] **T3** `hana.k@example.net`
- [x] **T4** `Actually, hold on - I need to check with my partner first`

### H09 — [92mPASS[0m  (edge)

_Party size of zero. It must never be stored or passed to a tool; the planner re-asks for a usable number._

- [x] **T1** `GoodFoods Nook`
- [x] **T2** `next friday, 0 guests`

### H10 — [92mPASS[0m  (edge)

_17:30 falls in the gap between lunch and dinner service, so no such slot exists on any day. The planner must still check rather than assume, and the conversation must NOT advance to booking on an unavailable slot._

- [x] **T1** `GoodFoods Tavern`
- [x] **T2** `5 people on 6 Feb at 5:30pm`
