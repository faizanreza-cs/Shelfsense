"""Dataiku Python recipe: drift and product alerts.

Inputs: products, orders, reviews, review_aspects, aspect_monthly
Outputs: drift_monthly, aspect_alerts
"""

import dataiku

from shelfsense.pipeline import run_monitoring

products = dataiku.Dataset("products").get_dataframe()
orders = dataiku.Dataset("orders").get_dataframe(parse_dates=True)
reviews = dataiku.Dataset("reviews").get_dataframe(parse_dates=True)
scored = {
    "review_aspects": dataiku.Dataset("review_aspects").get_dataframe(),
    "aspect_monthly": dataiku.Dataset("aspect_monthly").get_dataframe(),
}

_, tables = run_monitoring(products, orders, reviews, scored)
for name, df in tables.items():
    dataiku.Dataset(name).write_with_schema(df)
