-- =====================================================================
-- 05  One row per customer active at t0: three signals + path + outcome
-- =====================================================================
SELECT c.customer_id,
       c.customer_path,
       ch.channel_attempts_6m,
       ch.success_rate_raw,
       ch.success_rate_shrunk,
       ch.success_rate_naive,
       b.balance_trend,
       b.ever_overdrawn,
       v.txn_velocity,
       o.dormant_flag
FROM customers c
JOIN txn_velocity   v USING (customer_id)
LEFT JOIN channel_signal ch USING (customer_id)
LEFT JOIN balance_trend  b  USING (customer_id)
JOIN outcome        o USING (customer_id)
WHERE v.txn_last_3m > 0          -- active at t0
ORDER BY customer_id   -- deterministic row order (DuckDB is multi-threaded)
