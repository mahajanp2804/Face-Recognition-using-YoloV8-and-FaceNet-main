"""Attendance log: a ``Name,Timestamp`` CSV where each person is marked at most once a day."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

HEADER = ["Name", "Timestamp"]
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"


def attendance_exists(name: str, path: Path, now: datetime | None = None) -> bool:
    """Return True if ``name`` already has an entry dated today (``now``'s date) in ``path``."""
    if not path.exists():
        return False

    today = (now or datetime.now()).strftime(DATE_FORMAT)

    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) < 2:
                continue
            stored_name, timestamp = row[0], row[1]
            if stored_name == name and timestamp.startswith(today):
                return True

    return False


def mark_attendance(name: str, path: Path, now: datetime | None = None) -> str:
    """Append ``name`` with the current timestamp, writing the header if the file is new.

    Returns the timestamp that was written.
    """
    timestamp = (now or datetime.now()).strftime(TIMESTAMP_FORMAT)
    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADER)
        writer.writerow([name, timestamp])

    logger.info("Attendance marked for %s at %s", name, timestamp)
    return timestamp
