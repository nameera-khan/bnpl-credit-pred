import logging
import sys
import os
def setup_logging(level=logging.INFO):
	os.makedirs("logs", exist_ok=True)
	logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
	
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/pipeline.log"),
        ]
    )
