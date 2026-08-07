"""I/O subpackage — result serialisation and movie processing.

Public API
----------
.. autosummary::
    EventRecord
    ParticleRecord
    ResultsFile
    save_results
    load_results
    build_event_records
    build_particle_records
"""

from tandemmp.io.save import (
    EventRecord,
    ParticleRecord,
    ResultsFile,
    build_event_records,
    build_particle_records,
    load_results,
    save_results,
)

__all__ = [
    "EventRecord",
    "ParticleRecord",
    "ResultsFile",
    "build_event_records",
    "build_particle_records",
    "load_results",
    "save_results",
]
