-- =====================================================================
-- 01  Channel signal: digital transaction success rate (last 6 months)
-- ---------------------------------------------------------------------
-- Two fixes compared with a naive implementation:
--   1. VOLUME-WEIGHTED aggregation: SUM(successes)/SUM(attempts) across all
--      channels. A plain average of per-channel rates lets 1 failed USSD
--      attempt weigh as much as 200 mobile transfers.
--   2. BAYESIAN SHRINKAGE towards the path average, with a prior worth
--      {k} attempts. A new customer with 2/3 successes should not be scored
--      as a 67% success customer.
-- The naive rate is kept so the notebook can show the difference.
-- =====================================================================
WITH win AS (
    SELECT c.customer_id, c.customer_path, m.channel, m.attempts, m.successes
    FROM channel_monthly m
    JOIN customers c USING (customer_id)
    WHERE m.month_idx BETWEEN {t0} - 6 AND {t0} - 1
),
per_customer AS (
    SELECT customer_id, customer_path,
           SUM(attempts)  AS attempts,
           SUM(successes) AS successes
    FROM win GROUP BY customer_id, customer_path
),
naive AS (
    SELECT customer_id, AVG(rate) AS success_rate_naive
    FROM (SELECT customer_id, channel, SUM(successes) * 1.0 / SUM(attempts) AS rate
          FROM win GROUP BY customer_id, channel)
    GROUP BY customer_id
),
prior AS (
    SELECT customer_path, SUM(successes) * 1.0 / SUM(attempts) AS path_rate
    FROM per_customer GROUP BY customer_path
)
SELECT p.customer_id,
       p.attempts AS channel_attempts_6m,
       p.successes * 1.0 / p.attempts                                     AS success_rate_raw,
       (p.successes + {k} * pr.path_rate) / (p.attempts + {k})            AS success_rate_shrunk,
       n.success_rate_naive
FROM per_customer p
JOIN prior pr USING (customer_path)
JOIN naive n  USING (customer_id)
