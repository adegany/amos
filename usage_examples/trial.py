"""Run each half of an example against a fresh HTTP service on the same trial DB."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

from amos.http_api import AmosHTTPServer


@contextmanager
def service(database):
    server = AmosHTTPServer(("127.0.0.1", 0), str(database))
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def run_trial(first_session, next_session):
    with TemporaryDirectory() as directory:
        database = Path(directory) / "amos.sqlite3"
        with service(database) as url:
            first_session(url)
        with service(database) as url:
            next_session(url)
