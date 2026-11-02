"""Measurement protocols for the gates.

Separate from ``experiment/``, which holds run metadata and is deliberately
stdlib-only so a manifest can be read anywhere without pulling in NumPy or the
environment.  The code here measures the environment, so it has those
dependencies by nature and cannot live under that constraint.
"""
