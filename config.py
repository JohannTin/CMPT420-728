import os
from dotenv import load_dotenv

load_dotenv()

# Set sequence length of data to be observed for prediction
SEQ_LEN=10

PRICE_MODEL_PATH=os.environ.get("PRICE_MODEL_PATH")
