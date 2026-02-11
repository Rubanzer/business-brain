"""Watchdog-based file watcher for the incoming data directory."""

from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config.settings import settings


class IncomingFileHandler(FileSystemEventHandler):
    """Handles new CSV/XML files dropped into the watch directory."""

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() == ".csv":
            # TODO: dispatch to csv_loader.load_csv(path)
            print(f"[watcher] New CSV detected: {path.name}")
        elif path.suffix.lower() == ".xml":
            print(f"[watcher] New XML detected: {path.name}")


def start_watcher() -> Observer:
    """Start watching the configured incoming directory. Returns the Observer."""
    watch_path = Path(settings.watch_directory)
    watch_path.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    observer.schedule(IncomingFileHandler(), str(watch_path), recursive=False)
    observer.start()
    print(f"[watcher] Watching {watch_path.resolve()}")
    return observer
