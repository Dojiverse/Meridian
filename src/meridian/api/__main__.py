"""Run the demo interface:  python -m meridian.api

Serves the advisor blotter at http://127.0.0.1:8000 against an
in-memory demo store, one world per visitor. No database, no build
step, no npm.

Environment, for hosting:

    PORT                 port to listen on (Cloud Run sets this); 8000
    MERIDIAN_HOST        bind address; 127.0.0.1 locally, 0.0.0.0 in a
                         container or a Codespace
    MERIDIAN_ROOT_PATH   mount the app under a prefix, e.g. /blotter,
                         when hosting behind a path-based rewrite
"""

from __future__ import annotations

import os

import uvicorn

from meridian.api.app import create_app
from meridian.api.demo import build_demo_store


def main() -> None:
    app = create_app(
        build_demo_store, root_path=os.environ.get("MERIDIAN_ROOT_PATH", "")
    )
    uvicorn.run(
        app,
        host=os.environ.get("MERIDIAN_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
