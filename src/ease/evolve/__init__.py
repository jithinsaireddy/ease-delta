"""Self-evolution: improving after deployment without retraining the large model.

Four mechanisms, ordered from immediate to slow. Each is versioned and reversible, and each enters
the engine as a change to the `w:agg` input, so the dependency graph recomputes the cheap nodes and
never reruns the encoder.

    memory         a verified correction takes effect on the next query; it can be deleted again
    calibration    temperature and class bias are refitted on recent verified readings
    threshold      the confidence required to endorse an action tracks the observed error rate
    consolidation  the small refiner network is fine-tuned on accumulated feedback, and the result
                   is adopted only if it passes a regression gate

What never changes here is the 149M-parameter encoder. User facts are not written into its
weights; they stay in addressable records that can be inspected and withdrawn.
"""
