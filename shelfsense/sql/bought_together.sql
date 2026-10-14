-- Products bought together
-- Product pairs bought by the same shoppers, with lift: how many times more often the pair is
-- bought together than if the two products were bought independently.
WITH buyers AS (
    SELECT DISTINCT user_id, item_id FROM orders
),
item_buyers AS (
    SELECT item_id, COUNT(*) AS n FROM buyers GROUP BY item_id
),
pairs AS (
    SELECT a.item_id AS item_a, b.item_id AS item_b, COUNT(*) AS users
    FROM buyers a
    JOIN buyers b ON a.user_id = b.user_id AND a.item_id < b.item_id
    GROUP BY a.item_id, b.item_id
)
SELECT pa.title AS product_a,
       pb.title AS product_b,
       pa.category AS category_a,
       pb.category AS category_b,
       pairs.users,
       ROUND(pairs.users * (SELECT COUNT(DISTINCT user_id) FROM buyers) / (ia.n * ib.n), 2)
           AS lift
FROM pairs
JOIN item_buyers ia ON ia.item_id = pairs.item_a
JOIN item_buyers ib ON ib.item_id = pairs.item_b
JOIN products pa ON pa.item_id = pairs.item_a
JOIN products pb ON pb.item_id = pairs.item_b
ORDER BY pairs.users DESC
LIMIT 50
