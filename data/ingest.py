import sqlite3
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

TABLES = {
    "application_train": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv"
}

def load_all_to_sqlite(raw_dir="data/raw/home-credit-default-risk", dp_path = "data/db/homecredit.db"):
    os.makedirs(os.path.dirname(dp_path),exist_ok=True)
    conn = sqlite3.connect(dp_path)
    for table_name, file_name in TABLES.items():
        path = os.path.join(raw_dir,file_name)
        if not os.path.exists(path):
            logger.warning(f"Skipping missing file: {path}")
            continue
        df = pd.read_csv(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info(f"Loaded {table_name}: {len(df):,} rows")
    conn.close()
    logger.info("All tables loaded into SQLite")

if __name__ == "__main__":
    from utils.logging_config import setup_logging
    setup_logging()
    load_all_to_sqlite()
    
