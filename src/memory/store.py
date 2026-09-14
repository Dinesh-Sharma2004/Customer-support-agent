import sqlite3
import os
import json
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from src.config.settings import settings

class Message(BaseModel):
    role: str
    content: str
    timestamp: str

class SessionMemory:
    """
    SQLite backed session store. PII redaction would happen before storing.
    """
    def __init__(self, db_path: str = settings.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    updated_at TEXT,
                    messages TEXT
                )
            ''')
            conn.commit()

    def create_session(self, session_id: str):
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, created_at, updated_at, messages) VALUES (?, ?, ?, ?)",
                (session_id, now, now, json.dumps([]))
            )
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str):
        # NOTE: A real implementation should apply a PII redaction layer to `content` here before storage.
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT messages FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row:
                messages = json.loads(row[0])
            else:
                messages = []
                self.create_session(session_id)
                
            msg = Message(role=role, content=content, timestamp=datetime.utcnow().isoformat())
            messages.append(msg.model_dump())
            
            cursor.execute(
                "UPDATE sessions SET updated_at = ?, messages = ? WHERE session_id = ?",
                (datetime.utcnow().isoformat(), json.dumps(messages), session_id)
            )
            conn.commit()

    def get_history(self, session_id: str, limit: int = 5) -> list[Message]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT messages FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row:
                messages = json.loads(row[0])
                # Return the last `limit` messages
                return [Message(**m) for m in messages[-limit:]]
            return []

    def clear_session(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
