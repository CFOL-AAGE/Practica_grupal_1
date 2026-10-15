"""
Configuración común del proyecto. Todas las decisiones compartidas viven aquí.
"""

from pathlib import Path
from pyspark.sql import SparkSession

# --- Rutas ---
ROOT = Path(__file__).resolve().parent.parent      # raíz del repo
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                         # parquet originales de Hugging Face
CLEAN_PATH = DATA_DIR / "clean.parquet"            # salida de data.py
FEATURES_DIR = ROOT / "features"                   # un parquet por familia de features
LEXICAL_PATH = FEATURES_DIR / "lexical.parquet"    # familia 1: (id, tf, tf_nostop)
CHAR_PATH = FEATURES_DIR / "char.parquet"          # familia 2: (id, tf_char)

HF_REPO = "artem9k/ai-text-detection-pile"

# --- Reproducibilidad ---
SEED = 42

# --- Datos ---
SAMPLE_SIZE = None        # None = dataset completo
TUNING_FRACTION = 0.15    # fracción de train usada en CrossValidator
MIN_WORDS = 50            # textos más cortos se descartan
WINDOW_WORDS = 100        # ventana fija para neutralizar el atajo de la longitud (justificada en análisis exploratorio)
LABELS = {"human": 0, "ai": 1}

# --- Splits ---
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
LM_FIT_FRACTION = 0.30    # % de humanos de train reservados SOLO para el modelo de n-gramas

# --- Spark ---
def get_spark(app_name: str = "aage-p1", driver_memory: str = "8g") -> SparkSession:
    """Crea (o reutiliza) la sesión de Spark con la misma configuración para todos."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")                               # usa todos los cores del Mac
        .config("spark.driver.memory", driver_memory)     # ajustar a la RAM de cada uno
        .config("spark.sql.shuffle.partitions", "64")     # 200 por defecto es excesivo en local
        .getOrCreate()
    )