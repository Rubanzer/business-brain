"""Watchdog-based file watcher for the incoming data directory."""

import asyncio
import logging
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config.settings import settings

logger = logging.getLogger(__name__)


class IncomingFileHandler(FileSystemEventHandler):
    """Handles new CSV/XML files dropped into the watch directory."""

    def __init__(self, db_session_factory=None):
        super().__init__()
        self._session_factory = db_session_factory

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() == ".csv":
            logger.info("New CSV detected: %s", path.name)
            self._dispatch_csv(path)
        elif path.suffix.lower() == ".xml":
            logger.info("New XML detected: %s — XML ingestion not yet implemented", path.name)

    def _dispatch_csv(self, path: Path) -> None:
        from business_brain.ingestion.csv_loader import load_csv

        if self._session_factory is None:
            logger.warning("No DB session factory configured — skipping %s", path.name)
            return

        async def _run():
            async with self._session_factory() as session:
                count = await load_csv(path, session)
                logger.info("Loaded %d rows from %s", count, path.name)

        asyncio.run(_run())


def start_watcher(db_session_factory=None) -> Observer:
    """Start watching the configured incoming directory. Returns the Observer."""
    watch_path = Path(settings.watch_directory)
    watch_path.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    handler = IncomingFileHandler(db_session_factory=db_session_factory)
    observer.schedule(handler, str(watch_path), recursive=False)
    observer.start()
    logger.info("Watching %s", watch_path.resolve())
    return observer
