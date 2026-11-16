-- =====================================================================
-- 03  Transaction velocity: recent 3 months vs the months before
-- ---------------------------------------------------------------------
-- Log ratio of average monthly transactions (last 3 months vs earlier
-- months in the 12-month window). New customers only have the months since
-- opening, so averages are computed over the months they actually had.
-- =====================================================================
SELECT customer_id,
       LN( (AVG(CASE WHEN month_idx >= {t0} - 3 THEN txn_count END) + 1)
         / (COALESCE(AVG(CASE WHEN month_idx <  {t0} - 3 THEN txn_count END),
                     AVG(CASE WHEN month_idx >= {t0} - 3 THEN txn_count END)) + 1) ) AS txn_velocity,
       SUM(CASE WHEN month_idx >= {t0} - 3 THEN txn_count ELSE 0 END)               AS txn_last_3m
FROM txn_monthly
WHERE month_idx BETWEEN {t0} - 12 AND {t0} - 1
GROUP BY customer_id
