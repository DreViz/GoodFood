# app/api/utils/date_utils.py
"""
Shared date / time normalization for the GoodFoods agent.

This module is the SINGLE source of truth for converting messy user-supplied
date and time strings into the canonical forms the rest of the system expects:

  dates -> "YYYY-MM-DD"   (consumed by slot_manager.parse_opening_hours)
  times -> "HH:MM"        (24-hour, consumed by check_availability)

It is used by three layers (see EVAL_BUGS.md BUG-1):

  1. planner_agent.safe_extract_date   -> normalize_date_to_iso
  2. tool_calls._normalize_date        -> normalize_date_to_iso
  3. slot_manager.get_available_slots  -> normalize_date_to_iso (defensive)

And by the planner's deterministic interceptors:

  - extract_date_from_text  / extract_time_from_text pull a value out of a
    free-form user message (used by the Phase-2 collection guard and the
    manage-flow modify interceptor).

Design goals:
  - No dependency on app.* (pure + stdlib + python-dateutil) so it stays
    unit-testable in isolation, mirroring tests/eval/scorer.py.
  - Day-first interpretation ("18/11" -> 18 Nov, NOT 11 Aug) because the user
    base is India/+05:30 — explicitly called out in EVAL_BUGS.md.
  - Always return a canonical string or None — never raise, never return a
    half-parsed value. Callers treat None as "not understood".
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

try:
    from dateutil import parser as _dateutil_parser
except Exception:  # pragma: no cover - dateutil is in requirements.txt
    _dateutil_parser = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Weekday name -> weekday index (Monday=0). Handles full names and 3-letter
# abbreviations; both are matched case-insensitively.
WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "weds": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

# Month name -> month number. Full + abbreviated.
MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _today(today: Optional[datetime]) -> datetime:
    return today if today is not None else datetime.now()


def _next_weekday(after: datetime, target_idx: int) -> datetime:
    """Return the next date on/after `after` whose weekday == target_idx.

    "this friday" and "next friday" are treated identically: the next
    occurrence of that weekday strictly after today. (Calendar-strict "this
    vs next" disambiguation is beyond what the 4B model or the user intends,
    and either is acceptable to the slot manager.)
    """
    delta = (target_idx - after.weekday()) % 7
    if delta == 0:
        delta = 7  # strictly next occurrence, not today
    return after + timedelta(days=delta)


def _safe_date(year: int, month: int, day: int, today: datetime) -> Optional[str]:
    """Build an ISO date, rolling to next year if the date has already passed."""
    try:
        d = datetime(year, month, day).date()
    except ValueError:
        return None
    # If the resolved date is in the past, assume the user means next year.
    # (Restaurants are booked in the near future; a past date is almost
    # always a year-rollover artefact of DD-Mon without a year.)
    if d < today.date() and (today.year - year) <= 0:
        try:
            d = datetime(year + 1, month, day).date()
        except ValueError:
            return None
    return d.isoformat()


# ---------------------------------------------------------------------------
# Public: date normalization
# ---------------------------------------------------------------------------

def normalize_date_to_iso(value, today: Optional[datetime] = None) -> Optional[str]:
    """Normalize a date value to "YYYY-MM-DD" or return None.

    `value` may be:
      - already ISO ("2026-08-09")            -> passthrough
      - a relative term ("today","tomorrow")   -> resolved against `today`
      - "this friday" / "next monday"          -> next weekday occurrence
      - "18 Nov", "18th November", "Nov 18"    -> DD/Mon, current/next year
      - "18/11", "18-11"                       -> day-first (DD/MM)
      - "2025-12-05"                           -> ISO passthrough
      - free text containing one of the above  -> extracted then normalized

    Never raises. None means "could not interpret as a date".
    """
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    now = _today(today)

    # 1) Already ISO?
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d).date().isoformat()
        except ValueError:
            pass

    # 2) Relative keywords (scan the whole string so this works on free text).
    if re.search(r"\btomorrow\b", text):
        return (now + timedelta(days=1)).date().isoformat()
    if re.search(r"\btoday\b", text):
        return now.date().isoformat()
    if re.search(r"\bday after tomorrow\b", text):
        return (now + timedelta(days=2)).date().isoformat()

    # 3) "this/next <weekday>"
    m = re.search(r"\b(?:this|next|coming)\s+([a-z]+day)\b", text)
    if m and m.group(1) in WEEKDAYS:
        return _next_weekday(now, WEEKDAYS[m.group(1)]).date().isoformat()
    # bare "next monday" / "next fri" (abbreviated)
    m = re.search(r"\b(?:this|next|coming)\s+(mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)\b", text)
    if m and m.group(1) in WEEKDAYS:
        return _next_weekday(now, WEEKDAYS[m.group(1)]).date().isoformat()

    # 4) "DD <Month>"  (18 Nov, 18th November)
    m = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|"
        r"august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|"
        r"sep|sept|oct|nov|dec)\b",
        text,
    )
    if m:
        return _safe_date(now.year, MONTHS[m.group(2)], int(m.group(1)), now)

    # 5) "<Month> DD"  (Nov 18, November 18th)
    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|"
        r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?\b",
        text,
    )
    if m:
        return _safe_date(now.year, MONTHS[m.group(1)], int(m.group(2)), now)

    # 6) Numeric "DD/MM" or "DD-MM" (day-first — user base is India/+05:30).
    #    The required separator means a bare number ("for 3 people") cannot
    #    match here — the false-positive risk came only from dateutil's fuzzy
    #    mode, which is deliberately not used (see note below).
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", text)
    if m:
        day, mo = int(m.group(1)), int(m.group(2))
        if 1 <= day <= 31 and 1 <= mo <= 12:
            return _safe_date(now.year, mo, day, now)

    # No fuzzy fallback on purpose: dateutil.parse(..., fuzzy=True) treats a
    # bare integer ("for 3 people" -> day 3) as a date, which corrupts the
    # Phase-2 collection guard. The explicit rules above cover every form the
    # eval exercises; anything else returns None and the caller re-asks.

    return None


def extract_date_from_text(text: str, today: Optional[datetime] = None) -> Optional[str]:
    """Pull a single date mention out of free-form user text.

    Used by the planner's deterministic guards when the model produced a
    `reply` but we still need to know whether the user just stated a date
    (e.g. Phase-2 collection order, manage-flow modify interceptor).

    Thin wrapper around normalize_date_to_iso — same rules, accepts raw text.
    """
    if not text:
        return None
    return normalize_date_to_iso(text, today=today)


# ---------------------------------------------------------------------------
# Public: time normalization
# ---------------------------------------------------------------------------

def normalize_time(value) -> Optional[str]:
    """Normalize a time value to 24-hour "HH:MM" or return None.

    Accepts: "20:00", "19:30", "8pm" -> "20:00", "3:00am" -> "03:00",
    "7:30 pm", "8 pm". Returns None if the value isn't a recognizable time.
    """
    if value is None:
        return None

    v = str(value).strip().lower()
    if not v:
        return None

    # Already 24-hour: "20:00", "19:30", "7:30"
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", v)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return f"{h:02d}:{mn:02d}"
        return None

    # 12-hour with am/pm: "8pm", "3:00am", "7:30 pm", "12 am"
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", v)
    if m:
        h = int(m.group(1))
        mn = int(m.group(2)) if m.group(2) else 0
        ap = m.group(3)
        if ap == "pm" and h != 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return f"{h:02d}:{mn:02d}"
        return None

    # Bare hour with am/pm but extra text — try to pull just the time token.
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", v)
    if m:
        return normalize_time(m.group(0))

    return None


def extract_time_from_text(text: str) -> Optional[str]:
    """Pull a single time mention out of free-form user text -> "HH:MM" or None.

    Prefers an am/pm-anchored token, then falls back to a bare HH:MM.
    """
    if not text:
        return None
    t = str(text).lower()

    # "8pm", "3:00am", "7:30 pm"
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        return normalize_time(m.group(0))

    # "20:00", "19:30"  (24-hour, no am/pm)
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        return normalize_time(m.group(0))

    return None
