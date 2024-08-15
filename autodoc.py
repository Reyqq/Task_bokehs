import subprocess
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Путь к вашей документации
docs_path = "docs/source"
build_command = "sphinx-build docs/source docs/"


class DocHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.rst', '.md', '.py')):  # добавьте другие расширения, если нужно
            print(f"Изменен файл: {event.src_path}")
            self.rebuild_docs()

    def rebuild_docs(self):
        print("Пересборка документации...")
        try:
            subprocess.run(build_command, shell=True, check=True)
            print("Документация успешно пересобрана.")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при сборке документации: {e}")


if __name__ == "__main__":
    event_handler = DocHandler()
    observer = Observer()
    observer.schedule(event_handler, path=docs_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
