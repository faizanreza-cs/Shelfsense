-- Monthly sales by category
-- Orders, revenue and distinct shoppers per month and category.
SELECT DATE_TRUNC('month', o.order_ts) AS month,
       p.category,
       COUNT(*) AS orders,
       ROUND(SUM(o.price), 2) AS revenue,
       COUNT(DISTINCT o.user_id) AS users
FROM orders o
JOIN products p ON p.item_id = o.item_id
GROUP BY 1, 2
ORDER BY 1, 2
