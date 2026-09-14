# Document 1: Golden Evaluation Set

## 1. Introduction and Provenance
This document outlines the construction, annotation, and methodological rationale for the **AmazonHelp Customer Support Golden Evaluation Set** (`amazonhelp_golden_set_final.csv`). 

The source of truth for all support interactions is the massive `twcs/twcs.csv` dataset, which contains millions of multi-brand customer service interactions. From this corpus, we isolated interactions specifically authored by the `@AmazonHelp` handle responding to customer inquiries. 

The golden evaluation set consists of exactly **200 hand-labelled examples** strategically sampled from the available pool of Amazon Help tweets. 

## 2. Realistic Sampling Strategy
Instead of artificially balancing the dataset to have an equal number of examples for every intent (which misrepresents production traffic and creates a false sense of performance), the sampling strategy was explicitly designed to approximate **realistic support traffic**. 

### Rationale & Composition
- **Natural Imbalance:** The set deliberately reflects the natural distribution of queries (e.g., Delivery and Account issues occur much more frequently than obscure edge cases).
- **Hard/Ambiguous Cases:** We explicitly selected multi-issue requests, highly ambiguous short queries (e.g., "help me"), and nuanced edge cases to test the model's fail-closed routing logic.
- **Why this is useful:** A naturally imbalanced set with realistic edge cases is a far better proxy for real-world deployment than a perfectly balanced dataset. It accurately measures the true precision/recall trade-offs an operational system will face.

## 3. Data Leakage Prevention
To ensure absolute methodological rigor and zero data leakage:
1. The 200 `tweet_id`s present in `amazonhelp_golden_set_final.csv` were **completely locked and excluded** before any other datasets were generated.
2. Our data preparation pipeline (`scripts/prepare_data_splits.py`) strictly enforces this exclusion, dropping any overlapping `tweet_id` or identical `text_customer` before generating the `train`, `calibration`, `dev`, and `retrieval_corpus` sets.
3. The golden set is **never** used for model fitting, threshold selection, prompt tuning, retrieval index construction, or LLM-judge calibration.

## 4. Shared Annotation Taxonomy
Each record in the golden set is evaluated against a strict, shared taxonomy. We explicitly separate human-provided ground truth labels from all automated model-generated labels.

### Intent Taxonomy (6-Class)
1. **Order & Delivery Issues:** Missing packages, tracking inquiries, delays.
2. **Prime & Membership:** Subscription benefits, cancellations, renewals.
3. **Refunds & Financials:** Billing disputes, refund status, unauthorized charges.
4. **Returns & Exchanges:** Damaged items, return labels, replacements.
5. **Account Security & Private Support:** Locked accounts, password resets, compromised data.
6. **Other / Human Review:** Highly ambiguous, uninterpretable, or multi-intent queries requiring human intervention.

### Evaluation Labels
- **Ambiguity Detected (bool):** Is the request inherently unclear?
- **Escalation Requirement (enum):** `AUTO_HANDLE`, `CLARIFY`, or `ESCALATE`.
- **Evidence Relevance (0-2):** Not Relevant (0), Partially Relevant (1), Highly Relevant (2).
- **Answer Correctness (0-2):** Incorrect (0), Partially Correct (1), Correct (2).

## 5. Annotation Guidelines and Adjudication
Human annotators followed specific guidelines to ensure consistency:
1. **Primary Intent:** If multiple issues exist, the label reflects the most urgent transactional necessity (e.g., Security > Refunds > Delivery).
2. **Ambiguity Bias:** When in doubt between two intents, annotators default to `Other / Human Review` to err on the side of safety.
3. **Consistency & Adjudication:** Disagreements between annotators are mathematically measured using Cohen's and Fleiss' Kappa statistics (implemented in `src/annotation/metrics.py`). Where inter-annotator agreement falls below 0.7, a third senior adjudicator resolves the tie (Majority Vote), marking the record as `NEEDS_REVIEW` if a tie persists.

## 6. Limitations
While rigorous, this golden set has limitations:
- **Sample Size:** 200 samples are sufficient for a strong baseline but may not have high enough statistical power to evaluate deep sub-intents (e.g., differentiating between a stolen package and a delayed package).
- **Imbalance:** Because it reflects reality, classes like `Returns & Exchanges` have fewer examples, meaning per-class metrics on minor classes have wider confidence intervals. We do not view this as a flaw, but as a known characteristic of the operational domain.
