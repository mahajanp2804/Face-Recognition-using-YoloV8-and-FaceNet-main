import csv
from datetime import datetime

from facerec.attendance import HEADER, attendance_exists, mark_attendance

MORNING = datetime(2026, 3, 2, 9, 0, 0)
AFTERNOON = datetime(2026, 3, 2, 15, 30, 0)
NEXT_DAY = datetime(2026, 3, 3, 9, 0, 0)


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.reader(f))


def test_creates_file_with_header(tmp_path):
    path = tmp_path / "attendance.csv"

    timestamp = mark_attendance("Alice", path, now=MORNING)

    assert timestamp == "2026-03-02 09:00:00"
    assert read_rows(path) == [HEADER, ["Alice", "2026-03-02 09:00:00"]]


def test_header_written_only_once(tmp_path):
    path = tmp_path / "attendance.csv"

    mark_attendance("Alice", path, now=MORNING)
    mark_attendance("Bob", path, now=AFTERNOON)

    rows = read_rows(path)
    assert rows[0] == HEADER
    assert rows.count(HEADER) == 1
    assert [r[0] for r in rows[1:]] == ["Alice", "Bob"]


def test_no_file_means_not_marked(tmp_path):
    assert not attendance_exists("Alice", tmp_path / "attendance.csv", now=MORNING)


def test_marked_later_the_same_day(tmp_path):
    path = tmp_path / "attendance.csv"
    mark_attendance("Alice", path, now=MORNING)

    assert attendance_exists("Alice", path, now=AFTERNOON)


def test_other_people_not_affected(tmp_path):
    path = tmp_path / "attendance.csv"
    mark_attendance("Alice", path, now=MORNING)

    assert not attendance_exists("Bob", path, now=AFTERNOON)


def test_new_day_is_not_marked(tmp_path):
    path = tmp_path / "attendance.csv"
    mark_attendance("Alice", path, now=MORNING)

    assert not attendance_exists("Alice", path, now=NEXT_DAY)


def test_once_per_day_flow(tmp_path):
    """The check-then-mark sequence used by `facerec run` logs a person once per day."""
    path = tmp_path / "attendance.csv"

    for now in (MORNING, AFTERNOON, NEXT_DAY):
        if not attendance_exists("Alice", path, now=now):
            mark_attendance("Alice", path, now=now)

    assert read_rows(path)[1:] == [
        ["Alice", "2026-03-02 09:00:00"],
        ["Alice", "2026-03-03 09:00:00"],
    ]


def test_ignores_blank_and_short_rows(tmp_path):
    path = tmp_path / "attendance.csv"
    path.write_text("Name,Timestamp\n\nAlice\nBob,2026-03-02 08:00:00\n")

    assert not attendance_exists("Alice", path, now=MORNING)
    assert attendance_exists("Bob", path, now=MORNING)
