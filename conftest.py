"""
Project-root conftest.py.

Two jobs:
  1. pytest always adds this file's directory to sys.path, which anchors
     imports like `from pipeline.signals import stylometric` regardless
     of how/where pytest is invoked from.
  2. Registers the --run-integration flag. This MUST live in conftest.py --
     pytest_addoption is silently ignored if defined inside a test module
     (tests/test_signals.py), which is the bug that caused
     "ValueError: no option named '--run-integration'".
"""


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="run tests that hit the real Groq API",
    )
