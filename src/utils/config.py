import os

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR  = os.path.join(ROOT_DIR, "saved_model")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# Data
MTSAMPLES_PATH = os.path.join(DATA_DIR, "mtsamples.csv")

TARGET_CATEGORIES = [
    "Cardiovascular / Pulmonary",
    "Neurology",
    "Gastroenterology",
    "Orthopedic",
    "Radiology",
    "Discharge Summary",
]

# Model
MODEL_NAME  = "dmis-lab/biobert-base-cased-v1.2"
MAX_LEN     = 256
BATCH_SIZE  = 8
EPOCHS      = 4
LR          = 2e-5
SEED        = 42
TEST_SIZE   = 0.2
VAL_SIZE    = 0.5   