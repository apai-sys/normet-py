"""PyInstaller entry point for the normet GUI.

``freeze_support`` must run before anything else so that joblib/loky worker
processes re-spawned from the frozen binary bootstrap correctly instead of
opening another GUI window.
"""

import multiprocessing

multiprocessing.freeze_support()

from normet.gui import main  # noqa: E402

raise SystemExit(main())
