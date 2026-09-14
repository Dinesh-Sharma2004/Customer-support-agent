# Data And Task Definition

The notebook uses the Twitter Customer Support dataset (`twcs/twcs.csv`) and constructs AmazonHelp-like customer-brand pairs from these fields: `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, and `in_response_to_tweet_id`.

The notebook joins brand responses to inbound customer tweets by parsing `response_tweet_id`, renaming brand/customer fields, and merging brand replies with customer messages. It then keeps the most active responding brand author IDs. This is a practical heuristic; the docs should not claim perfect handle-level identity resolution unless separately verified.

Preprocessing removes URLs, mentions, hashtags, and punctuation-like characters, lowercases text, and tokenizes on whitespace. `clean_text_str` rejoins tokens for classifier input; token lists are used for BM25.

A usable training example is a customer text paired with a brand response long enough to support heuristic resolution extraction. The notebook labels training data with keyword/resolution heuristics, not with exhaustive human labels.

## Splits

`TRAIN`: Used to train the classifier and construct the RAG corpus.

`CALIBRATION`: Used to fit post-hoc probability calibration.

`VALIDATION`: Used for model comparison, calibration-method selection, retrieval evaluation, intent-aware retrieval threshold search, and automation-policy search.

`GOLDEN`: A locked 200-row CSV used only for final evaluation. It must not be used to tune models, thresholds, prompts, retrieval settings, or policy choices.

The notebook code uses a 70/15/15 train/calibration/validation split after excluding Golden leakage candidates and stratifies by heuristic intent labels.

## Golden Caveats

The artifact `artifacts/amazonhelp_golden_set_final.csv` has 200 rows and 26 columns. Its `primary_intent` distribution includes 198 valid taxonomy labels, one value of `en`, and one missing value. That means final metrics should either filter invalid labels, as the notebook code does, or explicitly report that two rows are not valid intent labels.

Next: [Intent taxonomy](03_intent_taxonomy.md).
