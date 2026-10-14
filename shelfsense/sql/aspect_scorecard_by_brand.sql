-- Aspect scorecard by brand
-- Net sentiment per brand and aspect from the scored reviews: (positive - negative) / mentions,
-- with the number of reviewers behind each figure. Low scores show what to fix first.
SELECT p.brand,
       a.aspect,
       COUNT(*) AS mentions,
       ROUND((COUNT_IF(a.label = 'positive') - COUNT_IF(a.label = 'negative')) / COUNT(*), 3)
           AS net_sentiment,
       COUNT(DISTINCT r.user_id) AS users
FROM review_aspects a
JOIN reviews r ON r.review_id = a.review_id
JOIN products p ON p.item_id = a.item_id
GROUP BY p.brand, a.aspect
ORDER BY p.brand, a.aspect
