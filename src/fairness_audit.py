"""
Fairness & uncertainty audit.

Every "gender and age detector" tutorial silently presents its output as
fact. In reality, published research (Buolamwini & Gebru, "Gender
Shades", FAT* 2018) found large accuracy gaps in commercial gender
classifiers across skin tone and sex, and age estimation is well known
to be less reliable on children, older adults, and non-Western training
distributions. Most public face-analysis models (this one included) are
trained on datasets that skew toward lighter skin tones and Western
imagery.

This module doesn't try to "fix" that with model changes -- that would
be overclaiming. Instead it does the honest, useful thing: surfaces
per-prediction uncertainty, aggregates it over the session, and reports
it to the viewer alongside the predictions, so results are presented
as estimates with a visible confidence signal rather than ground truth.
"""

from __future__ import annotations

LOW_AGREEMENT_THRESHOLD = 0.6
LOW_GENDER_CONF_THRESHOLD = 0.65
HIGH_AGE_STD_THRESHOLD = 6.0  # years; a wide spread means the age reading isn't stable

DISCLOSURE_TEXT = """
### Model limitations & fairness notes

- **Estimates, not facts.** Age and gender outputs are statistical
  estimates from a model trained on public face datasets. They can be
  wrong, and "gender" here means the model's binary classification of
  outward appearance -- it does not measure or infer a person's actual
  gender identity.
- **Known bias risk.** Published audits of commercial and open face-
  analysis systems (e.g. Buolamwini & Gebru, 2018) found materially
  higher error rates for darker-skinned individuals and for women
  relative to lighter-skinned individuals and men. This system uses a
  general-purpose public model and has not been independently audited
  for bias across skin tone, age group, or ethnicity -- treat outputs
  with proportionally more skepticism for any group the underlying
  training data likely under-represents.
- **Age estimation is inherently approximate.** Error margins reported
  in the literature for public age-estimation models are typically
  several years, and are wider at the extremes (young children, 65+).
- **What this app does about it:** every prediction carries a live
  confidence/agreement score (see the Uncertainty panel). Low-agreement
  predictions are flagged rather than silently displayed as if certain,
  and the session summary reports what fraction of predictions were
  low-confidence.
"""


def flag_prediction(gender_confidence: float, agreement_ratio: float, age_std: float = 0.0) -> dict:
    """Returns a small dict describing whether/why a prediction should be
    flagged as uncertain, for UI display.

    gender_confidence: the model's own average confidence for the winning
                        label (NOT how often frames agreed with each other --
                        see smoothing.py for why those are tracked separately).
    agreement_ratio:   fraction of recent frames that agreed on the winning label.
    age_std:            spread of raw age readings in the window; a wide spread
                        flags an unstable age estimate even if the average looks fine.
    """
    flags = []
    if agreement_ratio < LOW_AGREEMENT_THRESHOLD:
        flags.append("unstable across recent frames")
    if gender_confidence < LOW_GENDER_CONF_THRESHOLD:
        flags.append("low gender confidence")
    if age_std > HIGH_AGE_STD_THRESHOLD:
        flags.append("age estimate fluctuating")
    return {
        "uncertain": len(flags) > 0,
        "reasons": flags,
    }


def session_fairness_report(summary: dict) -> str:
    """Given a SessionLog.summary() dict, produce a short human-readable note."""
    total = summary.get("total_events", 0)
    low_conf = summary.get("low_confidence_events", 0)
    if total == 0:
        return "No predictions recorded yet."
    pct = round(100 * low_conf / total, 1)
    return (
        f"{low_conf} of {total} predictions ({pct}%) this session were flagged "
        f"as low-confidence (unstable across frames or low classifier confidence). "
        f"Treat any single prediction as an estimate, especially for groups the "
        f"underlying training data may under-represent."
    )
