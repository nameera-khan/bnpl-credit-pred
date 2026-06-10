import sqlite3
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_flat_feature_table(db_path="data/db/homecredit.db"):
    conn = sqlite3.connect(db_path)
    logger.info("Building feature table from database...")

    app = pd.read_sql("""
        SELECT
            SK_ID_CURR, TARGET,
            AMT_CREDIT, AMT_ANNUITY, AMT_INCOME_TOTAL, AMT_GOODS_PRICE,
            AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0)  AS credit_income_ratio,
            AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0) AS annuity_income_ratio,
            AMT_CREDIT  / NULLIF(AMT_GOODS_PRICE, 0)  AS credit_goods_ratio,
            DAYS_BIRTH    / -365.0 AS age_years,
            DAYS_EMPLOYED / -365.0 AS years_employed,
            CASE WHEN DAYS_EMPLOYED > 0 THEN 1 ELSE 0 END AS is_unemployed,
            REGION_POPULATION_RELATIVE, REGION_RATING_CLIENT,
            EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3,
            (COALESCE(EXT_SOURCE_1,0) + COALESCE(EXT_SOURCE_2,0) +
             COALESCE(EXT_SOURCE_3,0)) / 3.0 AS ext_source_mean,
            (CASE WHEN EXT_SOURCE_1 IS NULL THEN 1 ELSE 0 END +
             CASE WHEN EXT_SOURCE_2 IS NULL THEN 1 ELSE 0 END +
             CASE WHEN EXT_SOURCE_3 IS NULL THEN 1 ELSE 0 END) AS ext_source_nulls,
            CASE NAME_CONTRACT_TYPE WHEN 'Cash loans' THEN 1 ELSE 0 END AS is_cash_loan,
            CASE CODE_GENDER        WHEN 'M'          THEN 1 ELSE 0 END AS is_male,
            CASE NAME_EDUCATION_TYPE
                WHEN 'Higher education' THEN 3
                WHEN 'Incomplete higher' THEN 2
                WHEN 'Secondary / secondary special' THEN 1
                ELSE 0 END AS education_encoded,
            FLAG_DOCUMENT_3, FLAG_DOCUMENT_6, FLAG_MOBIL, FLAG_WORK_PHONE
        FROM application_train
    """, conn)

    bureau_agg = pd.read_sql("""
        SELECT SK_ID_CURR,
            COUNT(*) AS bureau_loan_count,
            SUM(CASE WHEN CREDIT_ACTIVE='Active' THEN 1 ELSE 0 END) AS bureau_active_loans,
            SUM(AMT_CREDIT_SUM) AS bureau_total_credit,
            SUM(AMT_CREDIT_SUM_DEBT) AS bureau_total_debt,
            SUM(CASE WHEN AMT_CREDIT_SUM_OVERDUE > 0 THEN 1 ELSE 0 END) AS bureau_overdue_count
        FROM bureau GROUP BY SK_ID_CURR
    """, conn)

    prev_agg = pd.read_sql("""
        SELECT SK_ID_CURR,
            COUNT(*) AS prev_app_count,
            SUM(CASE WHEN NAME_CONTRACT_STATUS='Approved'  THEN 1 ELSE 0 END) AS prev_approved,
            SUM(CASE WHEN NAME_CONTRACT_STATUS='Refused'   THEN 1 ELSE 0 END) AS prev_refused,
            SUM(CASE WHEN NAME_CONTRACT_STATUS='Approved'  THEN 1 ELSE 0 END) * 1.0 /
                NULLIF(COUNT(*), 0) AS prev_approval_rate
        FROM previous_application GROUP BY SK_ID_CURR
    """, conn)

    install_agg = pd.read_sql("""
        SELECT SK_ID_CURR,
            COUNT(*) AS install_count,
            SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END) AS install_late_count,
            AVG(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT) AS install_avg_delay_days
        FROM installments_payments GROUP BY SK_ID_CURR
    """, conn)

    conn.close()

    df = app
    df = df.merge(bureau_agg,  on="SK_ID_CURR", how="left")
    df = df.merge(prev_agg,    on="SK_ID_CURR", how="left")
    df = df.merge(install_agg, on="SK_ID_CURR", how="left")

    agg_cols = [c for c in df.columns if c.startswith(("bureau_","prev_","install_"))]
    df[agg_cols] = df[agg_cols].fillna(0)

    logger.info(f"Feature table ready: {len(df):,} rows x {df.shape[1]} columns")
    return df

def get_feature_cols(df):
    exclude = ["SK_ID_CURR", "TARGET"]
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
