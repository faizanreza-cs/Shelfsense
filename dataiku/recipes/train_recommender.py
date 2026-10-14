"""Dataiku Python recipe: evaluate the recommenders and recommend for every shopper.

Inputs: products, orders, reviews
Outputs: recommendations (user_id, rank, item_id), recommender_eval (one row per model)
"""

import dataiku
import pandas as pd

from shelfsense.pipeline import run_recommender

products = dataiku.Dataset("products").get_dataframe()
orders = dataiku.Dataset("orders").get_dataframe(parse_dates=True)
reviews = dataiku.Dataset("reviews").get_dataframe(parse_dates=True)

result, recommendations, _ = run_recommender(products, orders, reviews)
dataiku.Dataset("recommendations").write_with_schema(recommendations)
dataiku.Dataset("recommender_eval").write_with_schema(pd.DataFrame(result["models"]))
