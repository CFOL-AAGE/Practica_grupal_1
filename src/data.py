"""
Carga, limpieza y partición del dataset. Salida: data/clean.parquet
"""

from huggingface_hub import snapshot_download
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import HF_REPO, RAW_DIR, get_spark

HEADING = r'''(?m)^[^\n]{0,150}[^.!?"'\n ][ \t]*$'''   # línea corta sin puntuación final


def download_raw() -> None:
    """Descarga los parquet originales de Hugging Face (solo la primera vez)."""
    if RAW_DIR.exists() and any(RAW_DIR.rglob("*.parquet")):
        print(f"Datos ya descargados en {RAW_DIR}")
        return
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns="data/*.parquet",
        local_dir=RAW_DIR,
    )
    print(f"Descarga completa en {RAW_DIR}")


def load_raw(spark: SparkSession) -> DataFrame:
    """Lee los 7 parquet como un único DataFrame."""
    return spark.read.parquet(str(RAW_DIR / "data"))

def add_subsource(df: DataFrame) -> DataFrame:
    """Fuente original inferida a partir de los rangos de id. NUNCA usar como feature."""
    i, h = F.col("id"), F.col("source") == "human"
    sub = (F.when(h & (i < 127_500), "human_ivypanda")
            .when(h & (i < 387_000), "human_webtext")
            .when(h & (i < 444_000), "human_hc3")
            .when(h, "human_reddit_wp")
            .when(i < 1_081_500, "ai_gpt3")
            .when(i < 1_339_500, "ai_gpt2")
            .when(i < 1_386_000, "ai_gptj")
            .otherwise("ai_chatgpt"))
    return df.withColumn("subsource", sub)


def clean_text(df: DataFrame) -> DataFrame:
    """Elimina artefactos de formato. Misma regla para ambas clases (no usa la etiqueta)."""
    t = F.col("text")
    # 1. Comillas LaTeX/tipográficas -> rectas
    t = F.regexp_replace(t, r"``|''|[“”]", '"')
    t = F.regexp_replace(t, r"[’‘]", "'")
    # 2. Títulos, índices y encabezados (antes de quitar los saltos de línea)
    t = F.regexp_replace(t, HEADING, "")
    # 3. Marcas de markdown
    t = F.regexp_replace(t, r"\*", "")
    # 4. Destokenizar: "can ' t", "do n't", "word ." , "( word )"
    t = F.regexp_replace(t, r"(\w) ' (\w)", "$1'$2")
    t = F.regexp_replace(t, r"(\w) (n't|'s|'re|'ve|'ll|'d|'m)\b", "$1$2")
    t = F.regexp_replace(t, r" ([,.!?;:)])", "$1")
    t = F.regexp_replace(t, r"\( ", "(")
    # 5. Espacios y saltos -> un espacio; quitar extremos
    t = F.trim(F.regexp_replace(t, r"\s+", " "))
    # 6. Empezar en la primera frase completa
    t = F.regexp_replace(t, r'^.*?[.!?]"? (?=[A-Z"])', "")
    return df.withColumn("text", t)


def audit_artifacts(df: DataFrame) -> None:
    """% de textos de cada clase con artefactos de formato (posibles atajos)."""
    text = F.col("text")
    checks = {
        "empieza_minuscula": text.rlike(r"^[a-z]"),
        "empieza_espacio": text.rlike(r"^\s"),
        "comillas_latex": text.contains("``"),
        "apostrofo_separado": text.rlike(r"\w [’'] \w"),
        "salto_literal": text.contains("\\n"),     # el texto "\n", no un salto real
        "salto_real": text.contains("\n"),
    }
    df.groupBy("source").agg(
        *[F.round(F.avg(cond.cast("int")) * 100, 2).alias(name) for name, cond in checks.items()]
    ).show(truncate=False)

    # Distribución de longitudes (percentiles 5, 25, 50, 75 y 95)
    df.groupBy("source").agg(
        F.percentile_approx(F.length(text), [0.05, 0.25, 0.5, 0.75, 0.95]).alias("percentiles_chars")
    ).show(truncate=False)


if __name__ == "__main__":
    download_raw()
    spark = get_spark()
    df = load_raw(spark)
    audit_artifacts(df)

    # Primera inspección
    # df.printSchema()
    # print("Filas totales:", df.count())
    # df.groupBy("source").count().show()
    # df.select(
    #     "source",
    #     F.length("text").alias("n_chars"),
    #     F.substring("text", 1, 120).alias("inicio"),
    # ).show(10, truncate=False)