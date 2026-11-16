-- =====================================================================
-- 04  Outcome for validation: did the customer go dormant?
-- ---------------------------------------------------------------------
-- Dormant = zero customer-initiated transactions in the final 3 months
-- (~90 days) of the ~11-month window after t0. Only used to VALIDATE the
-- score weights; the score itself never sees the outcome.
-- =====================================================================
SELECT customer_id,
       CASE WHEN SUM(CASE WHEN month_idx >= {end_month} - 2 THEN txn_count ELSE 0 END) = 0
            THEN 1 ELSE 0 END AS dormant_flag
FROM txn_monthly
WHERE month_idx BETWEEN {t0} AND {end_month}
GROUP BY customer_id
