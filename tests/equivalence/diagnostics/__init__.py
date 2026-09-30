"""Standalone diagnostics for the master-vs-develop benchmark.

Each module here answers ONE question about a built artifact and is run by hand
(``python -m tests.equivalence.diagnostics.<name> --help``) rather than by the
harness. They are checkers, not tests: they read finished networks and resource
files and write a PNG with its CSV twin, which is how a difference is signed off
(PROJECT.md section 3.2). They live in the repo so a diagnosis can be reproduced
after ``$SCRATCH`` is purged.
"""
