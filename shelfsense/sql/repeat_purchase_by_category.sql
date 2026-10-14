-- Repeat purchase rate by category
-- Share of a category's shoppers who bought the same product again, and their average orders
-- of it. Repeat buying is the clearest sign a product works for people.
WITH per_shopper AS (
    SELECT o.user_id, p.category, o.item_id, COUNT(*) AS orders
    FROM orders o
    JOIN products p ON p.item_id = o.item_id
    GROUP BY o.user_id, p.category, o.item_id
)
SELECT category,
       COUNT(DISTINCT user_id) AS users,
       ROUND(COUNT(DISTINCT IFF(orders > 1, user_id, NULL)) / COUNT(DISTINCT user_id), 4)
           AS repeat_rate,
       ROUND(AVG(orders), 2) AS orders_per_product
FROM per_shopper
GROUP BY category
ORDER BY repeat_rate DESC
