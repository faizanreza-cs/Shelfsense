-- Monthly aspect sentiment by category
-- Net sentiment per category, aspect and month. A category-level view of what shoppers say over
-- time; product-level spikes are in the monitoring alerts.
SELECT p.category,
       a.aspect,
       a.month,
       COUNT(*) AS mentions,
       ROUND((COUNT_IF(a.label = 'positive') - COUNT_IF(a.label = 'negative')) / COUNT(*), 3)
           AS net_sentiment,
       COUNT(DISTINCT r.user_id) AS users
FROM review_aspects a
JOIN reviews r ON r.review_id = a.review_id
JOIN products p ON p.item_id = a.item_id
GROUP BY p.category, a.aspect, a.month
ORDER BY p.category, a.aspect, a.month
