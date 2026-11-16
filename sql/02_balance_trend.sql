-- =====================================================================
-- 02  Balance trend (12 months), overdraft-aware
-- ---------------------------------------------------------------------
-- LN(balance) breaks on overdrawn (negative) balances, and dropping those
-- customers would remove exactly the ones in trouble. A signed log
--     slog(x) = SIGN(x) * LN(1 + |x|)
-- keeps them in and still compresses the heavy right tail.
-- The trend is the slope of slog(balance) per month.
-- =====================================================================
SELECT customer_id,
       REGR_SLOPE(SIGN(avg_balance) * LN(1 + ABS(avg_balance)), month_idx) AS balance_trend,
       MIN(avg_balance) < 0                                                AS ever_overdrawn,
       COUNT(*)                                                            AS balance_months
FROM balance_monthly
WHERE month_idx BETWEEN {t0} - 12 AND {t0} - 1
GROUP BY customer_id
HAVING COUNT(*) >= 3
