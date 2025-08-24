# app.py
import io
import re
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import tz
from dateutil.parser import parse as dtparse

import pandas as pd
import streamlit as st
from pdfminer.high_level import extract_text  # pdfminer.six

# =========================
# Page setup + Top banner
# =========================
st.set_page_config(page_title="UC Berkeley Class List → .ics", page_icon="📅", layout="centered")

# tighten top padding so the banner sits close to the top
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- Top-of-page instructions image (root folder) ---
BANNER_PATH = Path("instructions.png")
if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), use_column_width=True)
else:
    st.warning("instructions.png not found in the app root. Add it to show instructions at the top.")

# App title
st.title("UC Berkeley PDF → Google Calendar (.ics)")
st.caption("Upload your UC Berkeley 'Enrollment Center' PDF (multi-page). Lectures, Discussions (if scheduled), and Labs are separate events.")

# =========================
# Constants & helpers
# =========================
PT = tz.gettz("America/Los_Angeles")

DAY_MAP = {
    "Monday": "MO", "Tuesday": "TU", "Wednesday": "WE",
    "Thursday": "TH", "Friday": "FR", "Saturday": "SA", "Sunday": "SU",
}

ABBREV_CANON = {
    "M": "Monday", "Mon": "Monday",
    "T": "Tuesday", "Tu": "Tuesday", "Tue": "Tuesday", "Tues": "Tuesday",
    "W": "Wednesday", "Wed": "Wednesday",
    "R": "Thursday", "Th": "Thursday", "Thu": "Thursday", "Thur": "Thursday",
    "F": "Friday", "Fri": "Friday",
    "Sa": "Saturday", "Sat": "Saturday",
    "Su": "Sunday", "Sun": "Sunday",
    "R": "Thursday",
}

COURSE_CODE_RE = re.compile(r"^(?P<dept>[A-Z&]{2,})\s+(?P<num>\d{1,3}[A-Z]?)\b")
TITLE_LINE_RE  = re.compile(r"^[A-Z&]{2,}\s+\d{1,3}[A-Z]?\s+.+$")

SECTION_TYPES_LONG = [
    "Lecture", "Discussion", "Laboratory", "Lab", "Colloquium",
    "Seminar", "Studio", "Activity", "Workshop", "Tutorial",
    "Recitation", "Fieldwork"
]
SECTION_TYPES_SHORT = [
    "LEC", "DIS", "LAB", "SEM", "COL", "STU", "ACT", "WOR", "TUT", "REC", "FLD"
]

# Flexible section header
SECTION_RE = re.compile(
    rf"^(?P<stype>(?:{'|'.join(SECTION_TYPES_LONG + SECTION_TYPES_SHORT)}))"
    r"(?:\s*[:\-–—]?\s*(?P<snum>\d{3}))?"
    r"(?:\s*[:\-–—]?\s*(?P<class>\d{5}))?$",
    re.I
)

# “08/27/2025 - 12/12/2025”
DATE_RANGE_RE = re.compile(r"(\d{2}/\d{2}/\d{4})\s*[-–—]\s*(\d{2}/\d{2}/\d{4})")
DAYS_RE       = re.compile(r"^Days:\s*(.+)$", re.I)

# Times: tolerate “1 1:00AM”, capture any tail
TIMES_RE = re.compile(
    r"^Times:\s*"
    r"(?P<st>\d\s?\d:\d{2}\s*[AP]M|\d:\d{2}\s*[AP]M)\s*"
    r"(?:to|[-–—])\s*"
    r"(?P<en>\d\s?\d:\d{2}\s*[AP]M|\d:\d{2}\s*[AP]M)"
    r"(?P<tail>.*)$",
    re.I,
)

LOCATION_PREFIX_RE = re.compile(r"^Location:\s*(.+)$", re.I)
ROOM_PREFIX_RE     = re.compile(r"^Room:\s*(.+)$", re.I)
STATUS_RE          = re.compile(r"^Status:\s*(.+)$", re.I)
TBA_RE             = re.compile(r"^Schedule:\s*To\s*be\s*Announced$", re.I)

def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_text(t: str) -> str:
    return (
        t.replace("\u2013", "-").replace("\u2014", "-")
         .replace("\u00A0", " ").replace("\u2009", " ")
    )

def extract_text_from_pdf(uploaded_file) -> str:
    data = uploaded_file.read()
    with io.BytesIO(data) as f:
        txt = extract_text(f)  # all pages
    return normalize_text(txt)

def looks_like_course_title(line: str) -> bool:
    return bool(TITLE_LINE_RE.match(line.strip()))

def maybe_capture_title_block(lines, start_idx):
    # Capture wrapped titles (up to 2 continuation lines)
    base = clean_spaces(lines[start_idx])
    collected = [base]
    j = start_idx + 1
    while j < len(lines) and len(collected) < 3:
        nxt = lines[j].strip()
        if not nxt:
            break
        if SECTION_RE.match(nxt) or DAYS_RE.match(nxt) or DATE_RANGE_RE.search(nxt) or STATUS_RE.match(nxt) or ROOM_PREFIX_RE.match(nxt) or LOCATION_PREFIX_RE.match(nxt):
            break
        if re.fullmatch(r"[A-Z0-9&\-\.,\s/]+", nxt) and not nxt.startswith(("Units", "Grading", "Status", "Instructor")):
            collected.append(clean_spaces(nxt)); j += 1; continue
        break
    return clean_spaces(" ".join(collected)), j - 1

def _expand_runon_days(token: str):
    s = token.replace("TTh", "Tu Th").replace("TuTh", "Tu Th")
    if re.fullmatch(r"[MTWRFSU]+", s):
        s = " ".join("Th" if ch == "R" else ch for ch in s)
    out = []
    for t in re.split(r"[^\w]+", s):
        if not t: continue
        if t in ABBREV_CANON: out.append(ABBREV_CANON[t])
        elif t.capitalize() in DAY_MAP: out.append(t.capitalize())
    return out

def parse_days(days_text: str):
    s = days_text.strip()
    rng = re.match(
        r"^\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+to\s+"
        r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*$",
        s, re.I,
    )
    if rng:
        start_name, end_name = rng.group(1).capitalize(), rng.group(2).capitalize()
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        i, j = order.index(start_name), order.index(end_name)
        return order[i:j+1] if i <= j else order[i:] + order[:j+1]
    s = re.sub(r"[,\s/]+", " ", s)
    parts = [p for p in s.split(" ") if p]
    out = []
    for p in parts:
        cap = p.capitalize()
        if cap in DAY_MAP: out.append(cap); continue
        out.extend(_expand_runon_days(p))
    seen, dedup = set(), []
    for d in out:
        if d not in seen:
            seen.add(d); dedup.append(d)
    return dedup

def first_occurrence_on_or_after(start_date, weekday_idx):
    d = start_date
    while d.weekday() != weekday_idx:
        d += timedelta(days=1)
    return d

# repairs
def repair_time_token(tok: str) -> str:
    # "1 1:00AM" -> "11:00AM"
    return tok.replace(" ", "")

def repair_location_digits(s: str) -> str:
    # "Cory 1 11" -> "Cory 111" (collapse spaces inside digit runs)
    return re.sub(r"(?<=\d)\s+(?=\d)", "", s)

# =========================
# Section helpers
# =========================
def normalize_section_type(stype: str) -> str:
    stype = (stype or "").strip()
    upper = stype.upper()
    mapping = {
        "LEC": "Lecture", "DIS": "Discussion", "LAB": "Lab", "SEM": "Seminar",
        "COL": "Colloquium", "STU": "Studio", "ACT": "Activity", "WOR": "Workshop",
        "TUT": "Tutorial", "REC": "Recitation", "FLD": "Fieldwork",
    }
    if upper in mapping:
        return mapping[upper]
    for long in SECTION_TYPES_LONG:
        if long.lower() == stype.lower():
            return long
    return stype.title() if stype else stype

def pick_up_missing_numbers(lines, idx, current_num, current_class):
    # If section header omitted number/class, peek 1–2 lines ahead
    snum, sclass = current_num, current_class
    k = idx + 1
    for _ in range(2):
        if k >= len(lines): break
        nxt = lines[k].strip()
        mnum = re.match(r"^(\d{3})$", nxt)
        if mnum and not snum:
            snum = mnum.group(1)
        mcls = re.match(r"^(\d{5})$", nxt)
        if mcls and not sclass:
            sclass = mcls.group(1)
        both = re.match(r"^(\d{3})\s+(\d{5})$", nxt)
        if both:
            if not snum: snum = both.group(1)
            if not sclass: sclass = both.group(2)
        k += 1
    return snum, sclass

def looks_like_room_text(s: str) -> bool:
    # Heuristic for lines like "Cory 111", "Mulford 159", "Tan 180", "Wheeler 224", "Internet/Online"
    if not s: return False
    s = s.strip()
    if s.lower() in {"internet/online", "online", "tba", "to be announced"}:
        return True
    return bool(re.match(r"^[A-Za-z][A-Za-z &'/\-]*\s+\d[\d/]*[A-Za-z0-9\-]*$", s))

def collect_location(lines, start_idx):
    """
    Location can be:
      - "Location: <text>" or "Room: <text>"
      - unlabeled like "Cory 111", "Mulford 159", "Internet/Online"
      - two-line: join them
    """
    l = lines[start_idx].strip()

    # labeled
    m = LOCATION_PREFIX_RE.match(l) or ROOM_PREFIX_RE.match(l)
    if m:
        loc = clean_spaces(m.group(1))
        return repair_location_digits(loc), start_idx

    # unlabeled heuristic (skip meta)
    if (l
        and not l.startswith(("Days:", "Times:", "Status", "Grading", "Units", "Instructor", "Location:", "Room:", "Schedule:"))
        and not DATE_RANGE_RE.search(l)
        and looks_like_room_text(l)
    ):
        loc = clean_spaces(l)
        # Optional continuation (e.g., room number on next line)
        if start_idx + 1 < len(lines):
            nxt = lines[start_idx + 1].strip()
            if (nxt
                and not nxt.startswith(("Days:", "Times:", "Status", "Grading", "Units", "Instructor", "Location:", "Room:", "Schedule:"))
                and not DATE_RANGE_RE.search(nxt)
                and (looks_like_room_text(nxt) or len(nxt) <= 10)
            ):
                loc = clean_spaces(f"{loc} {nxt}")
                return repair_location_digits(loc), start_idx + 1
        return repair_location_digits(loc), start_idx

    return None, start_idx

# =========================
# Core parser (course-level section FIFO + hard de-dup + TBA skip)
# =========================
def find_blocks(text: str, enroll_filter: str = "enrolled_or_waitlisted"):
    lines = [ln.strip() for ln in text.splitlines()]
    blocks = []
    seen_keys = set()

    current_title = None
    current_course_code = None
    section_queue = []   # sections awaiting a date block for this course

    i = 0
    while i < len(lines):
        ln = lines[i]

        # Course title (with wrapped continuation)
        if looks_like_course_title(ln):
            full_title, last_i = maybe_capture_title_block(lines, i)
            current_title = full_title
            mcode = COURSE_CODE_RE.match(full_title)
            current_course_code = f"{mcode.group('dept')} {mcode.group('num')}" if mcode else None
            section_queue = []  # reset for the new course
            i = last_i + 1
            continue

        # Section header -> enqueue (Lecture/Discussion/Lab/etc.)
        msec = SECTION_RE.match(ln)
        if msec:
            stype = normalize_section_type(msec.group("stype"))
            snum  = msec.group("snum")
            scls  = msec.group("class")
            snum, scls = pick_up_missing_numbers(lines, i, snum, scls)
            section_queue.append({"type": stype, "num": snum, "class": scls, "status": None})
            i += 1
            continue

        # (optional) per-section Status
        mstatus = STATUS_RE.match(ln)
        if mstatus and section_queue:
            section_queue[-1]["status"] = clean_spaces(mstatus.group(1))
            i += 1
            continue

        # Date range -> assign to next section in FIFO queue
        mdr = DATE_RANGE_RE.search(ln)
        if mdr and section_queue:
            start_date, end_date = mdr.group(1), mdr.group(2)
            section = section_queue.pop(0)

            # Scan forward for Days/Times/Location; stop at clear boundaries
            j = i + 1
            days_text = None
            start_time = end_time = None
            location_candidate = None
            skip_block = False

            while j < len(lines) and j <= i + 30:
                l2 = lines[j]

                # Boundaries (new title, new section, or next date range)
                if looks_like_course_title(l2) or SECTION_RE.match(l2) or DATE_RANGE_RE.search(l2):
                    break

                # If the block is explicitly TBA, skip it entirely (e.g., Discussion TBA)
                if TBA_RE.match(l2):
                    skip_block = True
                    break

                md = DAYS_RE.match(l2)
                if md:
                    days_text = clean_spaces(md.group(1))

                mt = TIMES_RE.match(l2)
                if mt:
                    st_tok = repair_time_token(clean_spaces(mt.group("st").upper()))
                    en_tok = repair_time_token(clean_spaces(mt.group("en").upper()))
                    start_time, end_time = st_tok, en_tok

                    # tail rarely contains good location in this PDF, but keep for robustness
                    tail = clean_spaces(mt.group("tail") or "")
                    tail = tail.lstrip(" .,-–—")
                    if tail and not tail.lower().startswith("schedule:"):
                        location_candidate = repair_location_digits(tail)

                # labeled or unlabeled location (Room/Location column)
                if location_candidate is None:
                    loc, last_idx = collect_location(lines, j)
                    if loc:
                        location_candidate = loc
                        j = last_idx  # skip consumed line(s)

                # optional status inside block
                mstatus2 = STATUS_RE.match(l2)
                if mstatus2 and section_queue:
                    section["status"] = clean_spaces(mstatus2.group(1))

                j += 1

            # Apply filter (include when status missing)
            status_norm = (section.get("status") or "").lower().replace("-", "").replace(" ", "")
            if enroll_filter == "enrolled":
                keep = (status_norm == "" or status_norm.startswith("enrolled"))
            elif enroll_filter == "enrolled_or_waitlisted":
                keep = (status_norm == "" or status_norm.startswith("enrolled") or status_norm.startswith("waitlisted"))
            else:
                keep = True

            # Only create if not TBA and has days/times
            if keep and not skip_block and days_text and start_time and end_time:
                days_norm = " ".join(parse_days(days_text))
                key = (
                    current_course_code or "",
                    section["type"] or "",
                    section["num"] or "",
                    section["class"] or "",
                    start_date, end_date,
                    days_norm,
                    start_time.replace(" ", ""),
                    end_time.replace(" ", ""),
                    (location_candidate or "TBA").lower(),
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    blocks.append({
                        "title": current_title,
                        "course_code": current_course_code,
                        "section_type": section["type"],
                        "section_num": section["num"],
                        "class_number": section["class"],
                        "status": section.get("status"),
                        "start_date": start_date,
                        "end_date": end_date,
                        "days_text": days_text,
                        "start_time": start_time,
                        "end_time": end_time,
                        "location": location_candidate or "TBA",
                    })

            i = j
            continue

        i += 1

    return blocks

# =========================
# ICS building
# =========================
def build_vtimezone_block():
    return (
        "BEGIN:VTIMEZONE\n"
        "TZID:America/Los_Angeles\n"
        "X-LIC-LOCATION:America/Los_Angeles\n"
        "BEGIN:DAYLIGHT\n"
        "TZOFFSETFROM:-0800\n"
        "TZOFFSETTO:-0700\n"
        "TZNAME:PDT\n"
        "DTSTART:19700308T020000\n"
        "RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=2SU\n"
        "END:DAYLIGHT\n"
        "BEGIN:STANDARD\n"
        "TZOFFSETFROM:-0700\n"
        "TZOFFSETTO:-0800\n"
        "TZNAME:PST\n"
        "DTSTART:19701101T020000\n"
        "RRULE:FREQ=YEARLY;BYMONTH=11;BYDAY=1SU\n"
        "END:STANDARD\n"
        "END:VTIMEZONE\n"
    )

def escape_text(s: str) -> str:
    if s is None: return ""
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace(",", "\\,").replace(";", "\\;")

def first_occurrence_on_or_after(start_date, weekday_idx):
    d = start_date
    while d.weekday() != weekday_idx:
        d += timedelta(days=1)
    return d

def first_dt_on_or_after(start_date, day_name: str, start_time):
    wd = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_name)
    date0 = first_occurrence_on_or_after(start_date, wd)
    return datetime.combine(date0, start_time).replace(tzinfo=PT)

def blocks_to_dataframe(blocks):
    rows = []
    for b in blocks:
        base_title = b["title"] or ""
        section_display = f"{b['section_type']} {b['section_num']}" if b["section_type"] and b["section_num"] else ""
        class_id = f"(Class# {b['class_number']})" if b["class_number"] else ""
        display_title = f"{base_title} — {section_display} {class_id}".strip() if section_display else base_title
        rows.append({
            "Title": display_title,
            "Course Code": b["course_code"] or "",
            "Section Type": b["section_type"] or "",
            "Section Number": b["section_num"] or "",
            "Class Number": b["class_number"] or "",
            "Status": b["status"] or "",
            "Start Date": b["start_date"] or "",
            "End Date": b["end_date"] or "",
            "Days": b["days_text"] or "",
            "Start Time": b["start_time"] or "",
            "End Time": b["end_time"] or "",
            "Location": b["location"] or "",
        })
    return pd.DataFrame(rows)

def dataframe_to_ics(df: pd.DataFrame) -> str:
    cal = []
    cal.append("BEGIN:VCALENDAR")
    cal.append("PRODID:-//UCB Class PDF -> ICS//EN")
    cal.append("VERSION:2.0")
    cal.append("CALSCALE:GREGORIAN")
    cal.append(build_vtimezone_block().strip())

    for _, row in df.iterrows():
        title = str(row.get("Title") or "").strip()
        location = str(row.get("Location") or "").strip()
        days_text = str(row.get("Days") or "").strip()
        st_date_str = str(row.get("Start Date") or "").strip()
        en_date_str = str(row.get("End Date") or "").strip()
        st_time_str = str(row.get("Start Time") or "").strip()
        en_time_str = str(row.get("End Time") or "").strip()
        if not (title and days_text and st_date_str and en_date_str and st_time_str and en_time_str):
            continue

        start_date = dtparse(st_date_str).date()
        end_date = dtparse(en_date_str).date()
        start_time = dtparse(st_time_str.replace(" ", "")).time()
        end_time = dtparse(en_time_str.replace(" ", "")).time()

        day_names = parse_days(days_text)
        byday = [DAY_MAP[d] for d in day_names if d in DAY_MAP]
        if not byday:
            continue

        dtstart_local = min(first_dt_on_or_after(start_date, d, start_time) for d in day_names)
        dtend_local   = datetime.combine(dtstart_local.date(), end_time).replace(tzinfo=PT)
        until_utc     = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=PT).astimezone(tz.UTC)

        cal.append("BEGIN:VEVENT")
        cal.append(f"UID:{uuid.uuid4()}@ucb-pdf-to-ics")
        cal.append(f"SUMMARY:{escape_text(title)}")
        if location:
            cal.append(f"LOCATION:{escape_text(location)}")
        cal.append(f"DTSTART;TZID=America/Los_Angeles:{dtstart_local.strftime('%Y%m%dT%H%M%S')}")
        cal.append(f"DTEND;TZID=America/Los_Angeles:{dtend_local.strftime('%Y%m%dT%H%M%S')}")
        cal.append(f"RRULE:FREQ=WEEKLY;BYDAY={','.join(byday)};UNTIL={until_utc.strftime('%Y%m%dT%H%M%SZ')}")
        cal.append("END:VEVENT")

    cal.append("END:VCALENDAR")
    return "\r\n".join(cal) + "\r\n"

# =========================
# Main UI
# =========================
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

filter_choice = st.selectbox(
    "Include which sections?",
    ["Enrolled or Waitlisted", "Enrolled only", "All sections"],
    index=0,  # inclusive by default; if no per-section Status, we still include
)

if uploaded:
    with st.spinner("Parsing your PDF..."):
        try:
            text = extract_text_from_pdf(uploaded)
            mode = {
                "Enrolled only": "enrolled",
                "Enrolled or Waitlisted": "enrolled_or_waitlisted",
                "All sections": "all",
            }[filter_choice]

            blocks = find_blocks(text, enroll_filter=mode)
            if not blocks and mode != "all":
                blocks = find_blocks(text, enroll_filter="all")

            if not blocks:
                st.error("I couldn't find any class meetings in that PDF. Double-check it's the 'Enrollment Center' print view.")
                with st.expander("Show raw text (debug)"):
                    st.code(text[:12000] + ("..." if len(text) > 12000 else ""), language="text")
            else:
                df = blocks_to_dataframe(blocks)

                # Always deduplicate (no toggle)
                dedup_cols = [
                    "Course Code", "Section Type", "Section Number", "Class Number",
                    "Start Date", "End Date", "Days", "Start Time", "End Time", "Location"
                ]
                before = len(df)
                df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
                removed = before - len(df)
                if removed > 0:
                    st.caption(f"Removed {removed} duplicate meeting(s).")

                st.success(f"Found {len(df)} class meeting(s).")
                st.write("You can edit any titles or locations below before exporting:")
                editor = st.data_editor(df, use_container_width=True, num_rows="fixed", hide_index=True)

                if st.button("Generate .ics"):
                    ics_text = dataframe_to_ics(editor)
                    st.download_button(
                        label="Download .ics",
                        data=ics_text.encode("utf-8"),
                        file_name="ucb-classes.ics",
                        mime="text/calendar",
                    )

                with st.expander("Show raw text (debug)"):
                    st.code(text[:12000] + ("..." if len(text) > 12000 else ""), language="text")

        except Exception as e:
            st.exception(e)
else:
    st.info("Drop your PDF above. Tip: CalCentral → My Academics → Class Enrollment → print view → Save as PDF.")
