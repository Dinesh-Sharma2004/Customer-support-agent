from contextlib import contextmanager
from typing import Generator
from src.memory.store import SessionMemory

class SessionContextManager:
    def __init__(self, memory_store: SessionMemory):
        self.memory = memory_store

    @contextmanager
    def context(self, session_id: str, is_new: bool = False) -> Generator[SessionMemory, None, None]:
        """
        Explicit session control.
        If is_new is True, the previous context is deliberately cleared to guarantee isolation.
        """
        if is_new:
            self.memory.clear_session(session_id)
            self.memory.create_session(session_id)
            
        yield self.memory
