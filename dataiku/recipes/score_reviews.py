"""Dataiku Python recipe: review aspect sentiment.

Inputs: reviews, and optionally review_labels (review_id plus one column per aspect, e.g. from
`shelfsense label --labeller llm`)
Outputs: review_aspects, product_aspects, aspect_monthly

Without review_labels, a sample is labelled with the keyword rules. The student model then
scores every review.
"""

import dataiku

from shelfsense.pipeline import run_aspects

reviews = dataiku.Dataset("reviews").get_dataframe(parse_dates=True)
try:
    labels = dataiku.Dataset("review_labels").get_dataframe()
except Exception:  # the optional input isn't in the flow
    labels = None

_, tables = run_aspects(reviews, labels=labels)
for name, df in tables.items():
    dataiku.Dataset(name).write_with_schema(df)
