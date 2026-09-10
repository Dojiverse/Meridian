"""Run the demo interface:  python -m meridian.api

Serves the advisor blotter at http://127.0.0.1:8000 against the
in-memory demo store. No database, no build step, no npm.
"""

from __future__ import annotations

import uvicorn

from meridian.api.app import create_app
from meridian.api.demo import build_demo_store


def main() -> None:
    uvicorn.run(create_app(build_demo_store()), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
