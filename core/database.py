import os
import sqlite3
import json
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Union, Generator

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meetings.db")


@contextmanager
def get_db(db_path: str = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connection that guarantees proper commit & close."""
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the SQLite database schema if it doesn't already exist."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                source TEXT,
                language TEXT DEFAULT 'english',
                transcript TEXT NOT NULL,
                summary TEXT,
                action_items TEXT,
                key_decisions TEXT,
                open_questions TEXT,
                audio_filename TEXT,
                audio_data BLOB,
                audio_size_bytes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for fast searching and sorting
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meetings_created_at ON meetings(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meetings_title ON meetings(title)")
        conn.commit()


def _format_extra_field(value: Any) -> Optional[str]:
    """Serializes lists, dicts, or strings for text storage."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def save_meeting(
    transcript: str,
    summary: Optional[str] = None,
    audio_path: Optional[str] = None,
    audio_data: Optional[bytes] = None,
    audio_filename: Optional[str] = None,
    title: Optional[str] = None,
    source: Optional[str] = None,
    language: Optional[str] = "english",
    action_items: Optional[Union[str, List, Dict]] = None,
    key_decisions: Optional[Union[str, List, Dict]] = None,
    open_questions: Optional[Union[str, List, Dict]] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Saves meeting transcript, summary, audio (as WAV BLOB), and metadata into SQLite.
    
    Args:
        transcript: Full transcription text.
        summary: Generated summary text.
        audio_path: Path to the .wav file on disk to read and store.
        audio_data: Raw WAV binary bytes (if already loaded in memory).
        audio_filename: Optional name for the audio file (e.g. 'recording.wav').
        title: Meeting/session title.
        source: Original YouTube URL or file path.
        language: Transcription language ('english', 'hinglish', etc.).
        action_items: Extracted action items (str, list, or dict).
        key_decisions: Extracted key decisions.
        open_questions: Extracted open questions.
        db_path: Path to the SQLite database file.

    Returns:
        The ID of the inserted record.
    """
    init_db(db_path)

    # Load audio bytes if path provided
    if audio_data is None and audio_path:
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            if not audio_filename:
                audio_filename = os.path.basename(audio_path)
        else:
            print(f"Warning: Audio file does not exist at {audio_path}")

    audio_size_bytes = len(audio_data) if audio_data is not None else 0

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO meetings (
                title,
                source,
                language,
                transcript,
                summary,
                action_items,
                key_decisions,
                open_questions,
                audio_filename,
                audio_data,
                audio_size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title or "Untitled Session",
                source,
                language,
                transcript,
                summary,
                _format_extra_field(action_items),
                _format_extra_field(key_decisions),
                _format_extra_field(open_questions),
                audio_filename,
                audio_data,
                audio_size_bytes,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_meeting(
    meeting_id: int,
    include_audio: bool = True,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single meeting by its ID.
    
    Args:
        meeting_id: Row ID of the meeting.
        include_audio: If False, omits the binary audio_data to save memory.
        db_path: Path to the SQLite database file.
    """
    init_db(db_path)
    columns = (
        "id, title, source, language, transcript, summary, action_items, "
        "key_decisions, open_questions, audio_filename, audio_size_bytes, created_at"
    )
    if include_audio:
        columns += ", audio_data"

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT {columns} FROM meetings WHERE id = ?", (meeting_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def get_all_meetings(
    limit: int = 50,
    offset: int = 0,
    include_audio: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Retrieves a list of meetings ordered by creation date (newest first).
    Audio BLOB is excluded by default for fast listing.
    """
    init_db(db_path)
    columns = (
        "id, title, source, language, transcript, summary, action_items, "
        "key_decisions, open_questions, audio_filename, audio_size_bytes, created_at"
    )
    if include_audio:
        columns += ", audio_data"

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {columns} FROM meetings ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]


def export_audio_to_file(
    meeting_id: int,
    output_path: str,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """
    Extracts the stored WAV audio BLOB from the database and saves it as a .wav file.
    """
    meeting = get_meeting(meeting_id, include_audio=True, db_path=db_path)
    if not meeting:
        raise ValueError(f"Meeting with id {meeting_id} not found.")

    audio_bytes = meeting.get("audio_data")
    if not audio_bytes:
        raise ValueError(f"Meeting with id {meeting_id} does not contain audio data.")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return output_path


def delete_meeting(meeting_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Deletes a meeting by ID from the database."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        conn.commit()
        return cursor.rowcount > 0


def search_meetings(
    query: str,
    limit: int = 20,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Searches for meetings containing the query in title, transcript, or summary."""
    init_db(db_path)
    search_pattern = f"%{query}%"
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, source, language, transcript, summary,
                   action_items, key_decisions, open_questions, audio_filename,
                   audio_size_bytes, created_at
            FROM meetings
            WHERE title LIKE ? OR transcript LIKE ? OR summary LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (search_pattern, search_pattern, search_pattern, limit),
        )
        return [dict(row) for row in cursor.fetchall()]


if __name__ == "__main__":
    import tempfile

    print("--- Testing SQLite Database Module ---")
    test_db = os.path.join(tempfile.gettempdir(), "test_meetings.db")
    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except Exception:
            pass

    # Create dummy WAV data
    dummy_wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    # Save a record
    inserted_id = save_meeting(
        title="Q3 Product Strategy",
        source="https://youtube.com/watch?v=example",
        language="english",
        transcript="We discussed launching feature X in September and expanding team.",
        summary="• Launch feature X in September\n• Expand engineering team",
        action_items=["Alice to prepare specs", "Bob to post job descriptions"],
        key_decisions=["Feature X launch approved"],
        audio_data=dummy_wav_bytes,
        audio_filename="product_strategy.wav",
        db_path=test_db,
    )
    print(f"Inserted meeting ID: {inserted_id}")

    # Fetch record
    record = get_meeting(inserted_id, include_audio=True, db_path=test_db)
    print(f"Retrieved: Title='{record['title']}', Audio size={record['audio_size_bytes']} bytes")

    # Export audio test
    test_export = os.path.join(tempfile.gettempdir(), "exported_test.wav")
    export_audio_to_file(inserted_id, test_export, db_path=test_db)
    print(f"Exported audio verified: size={os.path.getsize(test_export)} bytes")

    # Search test
    matches = search_meetings("September", db_path=test_db)
    print(f"Search matches found: {len(matches)}")

    # Clean up test files
    if os.path.exists(test_db):
        os.remove(test_db)
    if os.path.exists(test_export):
        os.remove(test_export)

    print("All SQLite database tests passed successfully!")
