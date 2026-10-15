"""
Procesamiento de datos...
"""
import os

from clean_data import build_clean
from config import CLEAN_PATH, get_spark



def data_already_exists() -> bool:
    """
    Verifica si los datos ya existen.
    """
    return os.path.exists(CLEAN_PATH)

def main():
    # 1) Loading the cleaning data pipeline and executing it.
    spark = get_spark()

    if not data_already_exists(): # Save 5 minutes of time execution.
        build_clean(spark)

    # # 2) Loading the clean data and start building the n-grams of the texts.
    # df_clean = spark.read.parquet(str(CLEAN_PATH))
    # print("Filas totales:", df_clean.count())


if __name__ == "__main__":
    main()
