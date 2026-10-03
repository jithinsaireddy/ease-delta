"""Relation labels shared by every component.

The edge function answers one question about a (claim, evidence) pair:
what does this evidence establish about this claim?
"""

SUPPORTS = 0  # evidence establishes the claim
REFUTES = 1  # evidence establishes the claim is false
NEI = 2  # evidence does not settle the claim (not enough information)

# Supervision-only label for corpora that distinguish just "entailed" from "not entailed".
# It is never predicted; the loss marginalises over REFUTES and NEI.
NOT_SUPPORTS = 3

RELATION_NAMES = ("SUPPORTS", "REFUTES", "NEI")
NUM_RELATIONS = 3
