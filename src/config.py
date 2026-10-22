"""
Configuración común del proyecto. Todas las decisiones compartidas viven aquí.
"""

from pathlib import Path
from pyspark.sql import SparkSession



# Rutas
ROOT = Path(__file__).resolve().parent.parent      # raíz del repo
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                         # parquet originales de Hugging Face
CLEAN_PATH = DATA_DIR / "clean.parquet"            # salida de clean_data.py
FEATURES_DIR = ROOT / "features"                   # un parquet por familia de features
LEXICAL_PATH = FEATURES_DIR / "lexical.parquet"    # familia 1: (id, tf, tf_nostop)
CHAR_PATH = FEATURES_DIR / "char.parquet"          # familia 2: (id, tf_char)
LM_PATH = FEATURES_DIR / "lm.parquet"
STYLE_PATH = FEATURES_DIR / "style.parquet"
MODELS_DIR = ROOT / "models"                       # PipelineModel entrenados (uno por familia y modelo)
RESULTS_DIR = ROOT / "results"                     # métricas de evaluate.py (CSV)


HF_REPO = "artem9k/ai-text-detection-pile"


# LM config
SEED = 42
LM_DIR = DATA_DIR / "lm"           # aquí se guardan los conteos de n-gramas
LM_ORDER = 3                       # trigramas: 2 palabras de contexto
LM_MIN_COUNT = 2                   # n-gramas (n>1) vistos menos veces se ignoran al puntuar
BACKOFF_ALPHA = 0.4                # penalización del Stupid Backoff (Brants et al., 2007)


# Datos
TUNING_FRACTION = 0.15    # fracción de train usada en CrossValidator
WINDOW_WORDS = 100        # ventana fija para neutralizar el atajo de la longitud; los textos más cortos se descartan (justificada en análisis exploratorio)


# Splits
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
LM_FIT_FRACTION = 0.30    # % de humanos de train reservados SOLO para el modelo de n-gramas


# Spark
def get_spark(app_name: str = "aage-p1", driver_memory: str = "44g") -> SparkSession:
    """
    Crea (o reutiliza) la sesión de Spark con la misma configuración
    para todos.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[16]")                              # 16 cores SLURM (-c 16)
        .config("spark.driver.memory", driver_memory)     # 44g de 48G SLURM (resto para JVM overhead)
        .config("spark.driver.maxResultSize", "8g")       # evita OOM al recoger resultados grandes
        .config("spark.sql.shuffle.partitions", "48")     # ~3× cores, equilibrio entre paralelismo y overhead
        .getOrCreate()
    )
