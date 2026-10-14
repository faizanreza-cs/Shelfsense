# Power BI report kit

Everything needed to build the shelfsense report in Power BI Desktop: consumer insights from
reviews, model monitoring and the recommender.

## 1. Export the data

```bash
shelfsense build            # if you haven't already
shelfsense export-powerbi   # writes powerbi/data/*.csv
```

| Table | Grain | Use |
|---|---|---|
| `dim_product` | product | title, brand, category, price, margin, stock |
| `dim_month` | month | month table (`month_start` is a date) |
| `fact_orders_monthly` | month x product | orders, revenue, shoppers |
| `fact_aspect_monthly` | month x product x aspect | positive, negative and total mentions |
| `fact_product_aspects` | product x aspect | the same over all months |
| `fact_drift_monthly` | month x metric | PSI against the reference months, status |
| `fact_aspect_alerts` | month x product x aspect | monitoring alerts |
| `fact_recommended_items` | product | how often the recommender puts it in a shopper's top 10 |
| `fact_recommender_eval` | model | recall@10, NDCG@10, coverage, margin |
| `fact_aspect_model` | model x aspect | macro-F1 of the labeller and the student model |

No table has a row per shopper: orders and recommendations are aggregated before export.

## 2. Load and model

1. **Get data > Text/CSV** for each file. In Power Query, set `month_start` to Date.
2. **Mark `dim_month` as a date table** (Table tools > Mark as date table, `month_start`).
3. **Relationships** (single direction, many-to-one):
   - `fact_orders_monthly[item_id]`, `fact_aspect_monthly[item_id]`,
     `fact_product_aspects[item_id]`, `fact_aspect_alerts[item_id]` and
     `fact_recommended_items[item_id]` → `dim_product[item_id]`
   - `fact_orders_monthly[month]`, `fact_aspect_monthly[month]`, `fact_drift_monthly[month]`
     and `fact_aspect_alerts[month]` → `dim_month[month]`
4. **Measures:** paste [`measures.dax`](measures.dax), one block at a time.
5. **Theme:** View > Themes > Browse for themes > [`theme.json`](theme.json). The colours match
   the README charts.

## 3. Pages

| Page | Visuals |
|---|---|
| **Consumer insights** | Matrix: `dim_product[brand]` by aspect with `Net Sentiment`, conditional formatting on a diverging scale (orange negative, blue positive). Line: `Net Sentiment` by `dim_month[month_start]`, aspect on the legend. Table: products with `Mentions`, `Net Sentiment vs Category`. Slicers: category, brand. |
| **Product drill-through** | Drill through on `dim_product[item_id]`: aspect bars with `Net Sentiment` and `Category Net Sentiment`, monthly line, alerts table. |
| **Model monitoring** | Matrix: `fact_drift_monthly[metric]` by month with `Max PSI`, coloured by `Drift Status`. Table: `fact_aspect_alerts` with product title, aspect, `Alert Negative Share` vs `Baseline Negative Share`, z. Card: `Alerts`. |
| **Recommender** | Bars: `Recall at 10` and `Catalogue Coverage` by model. Treemap: `Share of Recommendations` by brand. Card: `Recommended Margin` for the hybrid with rules. |
| **Sales** | Line: `Revenue` by month, category on the legend. Card: `Revenue vs Last Year %`. Table: top products by `Orders`. |

## 4. Screenshots

Save page screenshots to `docs/img/powerbi_*.png` and link them from the main README.
