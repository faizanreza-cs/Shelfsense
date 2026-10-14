-- One-time Snowflake setup for shelfsense.
-- Run as a role that can create warehouses, databases and roles (for example ACCOUNTADMIN on a
-- trial account). Then:
--   pip install "shelfsense[snowflake]"
--   export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PASSWORD=...
--   SNOWFLAKE_ROLE=SHELFSENSE_LOADER SNOWFLAKE_WAREHOUSE=SHELFSENSE_WH shelfsense load
-- and point the API, the agent and Dataiku at the read-only SHELFSENSE_ANALYST role:
--   SHELFSENSE_BACKEND=snowflake SNOWFLAKE_ROLE=SHELFSENSE_ANALYST shelfsense ask "..."

CREATE WAREHOUSE IF NOT EXISTS SHELFSENSE_WH
    WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE;
CREATE DATABASE IF NOT EXISTS SHELFSENSE;
CREATE SCHEMA IF NOT EXISTS SHELFSENSE.STORE;

-- Loader: creates and replaces the tables (used by `shelfsense load`)
CREATE ROLE IF NOT EXISTS SHELFSENSE_LOADER;
GRANT USAGE ON WAREHOUSE SHELFSENSE_WH TO ROLE SHELFSENSE_LOADER;
GRANT USAGE ON DATABASE SHELFSENSE TO ROLE SHELFSENSE_LOADER;
GRANT USAGE, CREATE TABLE, CREATE STAGE, CREATE FILE FORMAT ON SCHEMA SHELFSENSE.STORE
    TO ROLE SHELFSENSE_LOADER;

-- Analyst: read-only. Use it for the API, the agent and Dataiku, so model-written SQL can
-- never change data even if it got past the SQL guard.
CREATE ROLE IF NOT EXISTS SHELFSENSE_ANALYST;
GRANT USAGE ON WAREHOUSE SHELFSENSE_WH TO ROLE SHELFSENSE_ANALYST;
GRANT USAGE ON DATABASE SHELFSENSE TO ROLE SHELFSENSE_ANALYST;
GRANT USAGE ON SCHEMA SHELFSENSE.STORE TO ROLE SHELFSENSE_ANALYST;
GRANT SELECT ON ALL TABLES IN SCHEMA SHELFSENSE.STORE TO ROLE SHELFSENSE_ANALYST;
GRANT SELECT ON FUTURE TABLES IN SCHEMA SHELFSENSE.STORE TO ROLE SHELFSENSE_ANALYST;

-- Replace <your_user> with your Snowflake user name
GRANT ROLE SHELFSENSE_LOADER TO USER <your_user>;
GRANT ROLE SHELFSENSE_ANALYST TO USER <your_user>;

-- Optional, after `shelfsense load` (Enterprise Edition): an aggregation policy. Under the
-- analyst role, queries on orders and recommendations must aggregate, and groups smaller than
-- 50 rows are folded into a remainder, so not even a query that slipped past the SQL guard can
-- list a shopper's purchases. Reviews stay readable (the agent quotes review text, which is
-- public on the store anyway; the guard keeps user_id out). The loader role is exempt. Not run
-- in CI.
-- CREATE AGGREGATION POLICY IF NOT EXISTS SHELFSENSE.STORE.MIN_GROUP_50
--     AS () RETURNS AGGREGATION_CONSTRAINT ->
--         CASE
--             WHEN CURRENT_ROLE() = 'SHELFSENSE_ANALYST'
--                 THEN AGGREGATION_CONSTRAINT(MIN_GROUP_SIZE => 50)
--             ELSE NO_AGGREGATION_CONSTRAINT()
--         END;
-- ALTER TABLE SHELFSENSE.STORE.ORDERS SET AGGREGATION POLICY SHELFSENSE.STORE.MIN_GROUP_50;
-- ALTER TABLE SHELFSENSE.STORE.RECOMMENDATIONS SET AGGREGATION POLICY SHELFSENSE.STORE.MIN_GROUP_50;
