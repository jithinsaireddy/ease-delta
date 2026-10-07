"""EASE-Delta: revision-aware decision computation.

    from ease import EvidenceReader, Tracker

Design rule used throughout the package:
    learned models handle uncertain interpretation of language;
    exact code handles identities, versions, logic, arithmetic and permissions.
"""

__version__ = "0.1.0"
__all__ = ["EvidenceReader", "Reading", "Tracker", "define", "__version__"]


def __getattr__(name):
    # imported on first use, so `import ease` stays light
    if name in ("EvidenceReader", "Reading", "Tracker", "define"):
        from ease import easy

        return getattr(easy, name)
    raise AttributeError(f"module 'ease' has no attribute {name!r}")
