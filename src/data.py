"""
Carga, limpieza y partición del dataset. Salida: data/clean.parquet
"""

from huggingface_hub import snapshot_download
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import (HF_REPO, RAW_DIR, CLEAN_PATH, WINDOW_WORDS, SEED,
                        SPLITS, LM_FIT_FRACTION, get_spark)

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


def apply_window(df: DataFrame, n_words: int) -> DataFrame:
    """Recorta cada texto a sus primeras n_words palabras; descarta los más cortos."""
    words = F.split("text", " ")
    return (df.withColumn("n_words_orig", F.size(words))
              .filter(F.col("n_words_orig") >= n_words)
              .withColumn("text", F.array_join(F.slice(words, 1, n_words), " ")))


def drop_duplicates(df: DataFrame) -> DataFrame:
    """Una copia por texto; elimina textos que aparecen con etiquetas distintas."""
    por_texto = Window.partitionBy("text")
    return (df.withColumn("n_labels", F.size(F.collect_set("label").over(por_texto)))
              .withColumn("rn", F.row_number().over(por_texto.orderBy("id")))
              .filter((F.col("n_labels") == 1) & (F.col("rn") == 1))
              .drop("n_labels", "rn"))


def assign_split(df: DataFrame) -> DataFrame:
    """Split determinista por hash del id: no depende del particionado (Tema 3, diap. 51)."""
    h = F.pmod(F.xxhash64("id", F.lit(SEED)), 100)
    h_lm = F.pmod(F.xxhash64("id", F.lit(SEED + 1)), 100)
    t = int(SPLITS["train"] * 100)
    v = t + int(SPLITS["val"] * 100)
    split = (F.when((h < t) & (F.col("label") == 0.0) & (h_lm < LM_FIT_FRACTION * 100), "lm_fit")
              .when(h < t, "train")
              .when(h < v, "val")
              .otherwise("test"))
    return df.withColumn("split", split)


def build_clean(spark: SparkSession) -> None:
    """Pipeline completo de preprocesado -> data/clean.parquet"""
    df = load_raw(spark)
    df = add_subsource(df)
    df = clean_text(df)
    df = apply_window(df, WINDOW_WORDS)
    df = df.withColumn("label", (F.col("source") == "ai").cast("double"))
    df = drop_duplicates(df)
    df = assign_split(df)
    (df.select("id", "text", "label", "source", "subsource", "n_words_orig", "split")
       .write.mode("overwrite")
       .partitionBy("split")
       .parquet(str(CLEAN_PATH)))
    


if __name__ == "__main__":
    download_raw()
    spark = get_spark()
    build_clean(spark)
    clean = spark.read.parquet(str(CLEAN_PATH))
    clean.groupBy("split", "source").count().orderBy("split", "source").show()

    # Primera inspección
    # df.printSchema()
    # print("Filas totales:", df.count())
    # df.groupBy("source").count().show()
    # df.select(
    #     "source",
    #     F.length("text").alias("n_chars"),
    #     F.substring("text", 1, 120).alias("inicio"),
    # ).show(10, truncate=False)