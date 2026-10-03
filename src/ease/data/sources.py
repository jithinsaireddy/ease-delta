"""Registry of Hugging Face corpora, with the licence of each recorded next to it.

Licences were read from the Hub on 2026-09-29. `trainable=False` means the corpus must not
contribute gradient updates to a model intended for unrestricted use; it may still be used
to *measure* the model.

Every mapper returns a dict with the same keys, or None to skip the row:
    claim     the statement being judged (NLI hypothesis)
    evidence  the passage judged against (NLI premise)
    label     ease.labels constant
    source    registry key
    group     identifier shared by rows that must stay in the same split / contrast set
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ease.labels import NEI, NOT_SUPPORTS, REFUTES, SUPPORTS

Row = dict
Mapper = Callable[[Row], Optional[Row]]

_NLI_INT = {0: SUPPORTS, 1: NEI, 2: REFUTES}  # entailment, neutral, contradiction
_NLI_STR = {"entailment": SUPPORTS, "neutral": NEI, "contradiction": REFUTES}
_VITC = {"SUPPORTS": SUPPORTS, "REFUTES": REFUTES, "NOT ENOUGH INFO": NEI}


def _clean(s) -> str:
    return " ".join(str(s).split()) if s is not None else ""


def _mk(claim, evidence, label, source, group, page=None) -> Optional[Row]:
    claim, evidence = _clean(claim), _clean(evidence)
    if not claim or not evidence or label is None:
        return None
    return {"claim": claim, "evidence": evidence, "label": int(label), "source": source, "group": str(group),
            "page": f"{source}:{page if page is not None else group}"}


def map_vitaminc(r: Row) -> Optional[Row]:
    return _mk(r.get("claim"), r.get("evidence"), _VITC.get(r.get("label")), "vitaminc", r.get("case_id"),
               page=r.get("page"))


def map_mnli(r: Row) -> Optional[Row]:
    return _mk(r.get("hypothesis"), r.get("premise"), _NLI_INT.get(r.get("label")), "mnli", r.get("pairID"),
               page=r.get("promptID"))


def map_snli(r: Row) -> Optional[Row]:
    # SNLI marks no-consensus rows with label -1; _NLI_INT.get returns None and the row is skipped.
    return _mk(r.get("hypothesis"), r.get("premise"), _NLI_INT.get(r.get("label")), "snli", r.get("premise"))


def map_wanli(r: Row) -> Optional[Row]:
    return _mk(r.get("hypothesis"), r.get("premise"), _NLI_STR.get(r.get("gold")), "wanli", r.get("pairID"))


def map_anli(r: Row) -> Optional[Row]:
    return _mk(r.get("hypothesis"), r.get("premise"), _NLI_INT.get(r.get("label")), "anli", r.get("uid"))


def map_docnli(r: Row) -> Optional[Row]:
    lab = {"entailment": SUPPORTS, "not_entailment": NOT_SUPPORTS}.get(r.get("label"))
    return _mk(r.get("hypothesis"), r.get("premise"), lab, "docnli", r.get("premise", "")[:64])


@dataclass(frozen=True)
class SourceSpec:
    key: str
    hf_id: str
    config: Optional[str]
    train_split: Optional[str]
    eval_splits: tuple[str, ...]
    mapper: Mapper
    license: str
    trainable: bool
    note: str = ""
    columns: tuple = ()  # raw columns the mapper needs; others are dropped before rows are buffered


SOURCES: dict[str, SourceSpec] = {
    "vitaminc": SourceSpec(
        "vitaminc", "tals/vitaminc", None, "train", ("validation", "test"), map_vitaminc,
        "cc-by-sa-3.0", True,
        "Contrastive claim/evidence pairs built from real Wikipedia revisions (Schuster et al., NAACL 2021). "
        "The train file is ORDERED: 248,953 real-revision rows, then 121,700 synthetic rows with no NEI label. "
        "It must be shuffled across the whole file (see ease/data/streams.py).",
        columns=("claim", "evidence", "label", "case_id", "page", "revision_type"),
    ),
    "mnli": SourceSpec(
        "mnli", "nyu-mll/multi_nli", None, "train", ("validation_matched", "validation_mismatched"), map_mnli,
        "cc-by-3.0 / cc-by-sa-3.0 / mit / other (per genre)", True,
        "Multi-genre NLI (Williams et al., 2018).",
        columns=("premise", "hypothesis", "label", "pairID", "promptID", "genre"),
    ),
    "snli": SourceSpec(
        "snli", "stanfordnlp/snli", None, "train", ("validation", "test"), map_snli,
        "cc-by-sa-4.0", True,
        "Image-caption NLI (Bowman et al., 2015). Annotators assumed premise and hypothesis describe the same "
        "scene, so unrelated content is labelled contradiction. That convention is wrong for an evidence "
        "ledger (measured: scratch/probe_unrelated.py), so SNLI is evaluated on but not trained on.",
    ),
    "wanli": SourceSpec(
        "wanli", "alisawuffles/WANLI", None, "train", ("test",), map_wanli,
        "cc-by-4.0", True, "Worker-and-AI collaborative NLI (Liu et al., 2022).",
        columns=("premise", "hypothesis", "gold", "pairID", "genre"),
    ),
    "anli": SourceSpec(
        "anli", "facebook/anli", None, None, ("test_r1", "test_r2", "test_r3"), map_anli,
        "cc-by-nc-4.0", False,
        "Adversarial NLI. NON-COMMERCIAL licence: evaluation only, never trained on.",
    ),
    "docnli": SourceSpec(
        "docnli", "tasksource/doc-nli", None, "train", ("validation",), map_docnli,
        "bsd", True, "Document-level binary NLI; long premises. Optional, off by default.",
    ),
}


def get_source(key: str) -> SourceSpec:
    if key not in SOURCES:
        raise KeyError(f"unknown source {key!r}; known: {sorted(SOURCES)}")
    return SOURCES[key]
