"""
Familia 4 — Predictibilidad: modelo de lenguaje de n-gramas (Stupid Backoff)
entrenado con textos humanos (split lm_fit) y features de previsibilidad por texto.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

from config import CLEAN_PATH, LM_DIR, LM_ORDER, LM_MIN_COUNT, BACKOFF_ALPHA, FEATURES_DIR, SEED, get_spark

# Una palabra (con su contracción si la tiene: don't, it's) o un signo de puntuación suelto
TOKEN_RE = r"[a-z0-9]+(?:'[\p{L}0-9]+)?|[^a-z0-9\s]"
PAD = "<PAD>"


def tokenize(df: DataFrame) -> DataFrame:
    """
    Añade la columna `tokens`: texto en minúsculas troceado en palabras y puntuación,
    precedido de dos marcas de inicio <PAD>.
    """
    tokens = F.regexp_extract_all(F.lower("text"), F.lit(TOKEN_RE), 0)
    pad = F.array(*[F.lit(PAD)] * (LM_ORDER - 1)) # adición de padding al inicio de la oración
    return df.withColumn("tokens", F.concat(pad, tokens))


def _ngrams(tokens, order):
    """
    Todos los n-gramas de 1 a `order` palabras de una lista de tokens, como strings.
    """
    resultado = []
    for n in range(1, order + 1):                 # n = 1, 2, 3  (tamaño del grupo)
        for i in range(len(tokens) - n + 1):      # i = posición donde empieza el grupo
            grupo = tokens[i:i + n]               # coge n tokens seguidos desde la posición i
            resultado.append(" ".join(grupo))     # los une con espacios en un solo string
    return resultado


def count_ngrams(spark):
    """
    Sirve para tener una referencia de cómo es el lenguaje humano normal y medir cuánto se ajusta cada texto a ella.
    Se usan los textos humanos porque generalizan mejor: un modelo aprendido de textos de IA aprendería el estilo concreto de GPT-2, GPT-3, GPT-J y ChatGPT
    """
    # 1. Coge los textos humanos de lm_fit y los trocea en tokens
    lm_texts = tokenize(spark.read.parquet(str(CLEAN_PATH)).filter("split = 'lm_fit'"))

    # 2. De cada texto saca TODOS sus grupos de 1, 2 y 3 palabras (lo mismo que _ngrams, pero
    #    dentro de la JVM: sin workers de Python ni serializar cada n-grama a Python y de vuelta)
    grams = F.flatten(F.array(*[
        F.expr(f"transform(sequence(0, size(tokens) - {n}), i -> array_join(slice(tokens, i + 1, {n}), ' '))")
        for n in range(1, LM_ORDER + 1)]))

    # 3. WORDCOUNT: una fila por n-grama y se cuentan los iguales:  ("on the other", 5000)
    #    groupBy agrega parcialmente en cada partición y vuelca a disco si no cabe en memoria
    counts = (lm_texts.select(F.explode(grams).alias("ngram"))
              .groupBy("ngram").agg(F.count("*").alias("count")))

    # 4. Apunta si es de 1, 2 o 3 palabras (columna n) y lo guarda
    (counts.withColumn("n", F.size(F.split("ngram", " ")))
           .write.mode("overwrite").partitionBy("n").parquet(str(LM_DIR / "ngram_counts")))


def build_table(spark: SparkSession) -> None:
    """
    Añade a cada n-grama su contexto y su puesto entre las continuaciones de ese contexto,
    y descarta bigramas/trigramas vistos menos de LM_MIN_COUNT veces.
    """
    words = F.split("ngram", " ")
    t = (spark.read.parquet(str(LM_DIR / "ngram_counts"))
         .withColumn("ctx", F.array_join(F.slice(words, 1, F.col("n") - 1), " "))  # el contexto son todas las palabras menos la última
         .withColumn("rank", F.row_number().over( # agrupa los n-gramas que tienen el mismo contexto, ordena de más a menos frecuente y enumera
             Window.partitionBy("n", "ctx").orderBy(F.desc("count"), "ngram")))       # puesto dentro del contexto de mayor a menor (si hay empate, alfabéticamente)
         # El puesto se calcula antes de podar, así que refleja la posición real entre todas las continuaciones, incluidas las raras.
         .filter((F.col("n") == 1) | (F.col("count") >= LM_MIN_COUNT)))              # poda, bi/trigramas vistos solo una vez se descartan
    t.write.mode("overwrite").partitionBy("n").parquet(str(LM_DIR / "ngram_table"))


def _load_tables(spark: SparkSession):
    """
    Lee la tabla del modelo y la separa por tamaño, con claves numéricas (hash) para los joins.
    - k = key (clave): el n-grama convertido en número, para buscarlo.
        -> trabajamos con números por que comparar dos números es una sola operación; comparar dos textos es ir letra a letra.
            además, un número siempre ocupa 8bytes, un texto no.
    - c = count (conteo): cuántas veces aparece en los textos humanos.
    - r = rank (puesto): en qué posición queda entre las opciones de su contexto.
    """
    t = spark.read.parquet(str(LM_DIR / "ngram_table"))
    k = F.xxhash64("ngram")
    t3 = t.filter("n = 3").select(k.alias("k_uvw"), F.col("count").alias("c_uvw"), F.col("rank").alias("r_uvw"))
    t2 = t.filter("n = 2").select(k.alias("k"), "count", "rank")
    t1 = t.filter("n = 1").select(k.alias("k"), "count", "rank")
    stats = t.filter("n = 1").agg(F.sum("count").alias("N"), F.count("*").alias("V")).first()
    return t3, t2, t1, stats["N"], stats["V"]

# ---- Score tokens -------------
def token_rows(docs: DataFrame) -> DataFrame:
    """
    Una fila por palabra a puntuar (w), con sus dos palabras anteriores (u, v).
    """
    return (tokenize(docs) # añade columna de tokens
            .select("id", "tokens", F.posexplode("tokens").alias("pos", "w")) # posexplode-> explota la lista, es decir, crea una fila por token y dice en qué posición estaba
            .filter(F.col("pos") >= 2) # quita filas 0 y 1 -> las de PAD
            .withColumn("u", F.element_at("tokens", F.col("pos") - 1)) # saca de la lista la palabra que está dos posiciones antes y la llama u
            .withColumn("v", F.element_at("tokens", F.col("pos"))) # lo mismo, pero para la palabra justo anterior, que llamamos v
            .select("id", "pos", "u", "v", "w"))


def lookup_counts(spark: SparkSession, rows: DataFrame):
    """
    Para puntuar la palabra w necesitamos saber cuántas veces vio el modelo ciertas combinaciones.
    Esta función solo busca esos números en la tabla. Todavía no calcula nada.

    Usamos broadcast join por los siguientes motivos:
        - Tenemos una tabla enorme (dividida en núcleos), las ~140 M palabras a puntuar, y otra pequeña, el modelo de lenguaje.
        - Para hacer el join, Spark tiene que juntar las dos. Mover la grande es lentísimo; copiar la pequeña a cada núcleo es rápido.
            Así cada núcleo busca sus palabras sin mover nada.
    """
    t3, t2, t1, N, V = _load_tables(spark)
    B = F.broadcast

    # 1. Cada combinación se convierte en su número (igual que en la tabla)
    keys = rows.select("id", "pos", "u", "v", "w",
        F.xxhash64(F.concat_ws(" ", "u", "v", "w")).alias("k_uvw"),
        F.xxhash64(F.concat_ws(" ", "u", "v")).alias("k_uv"),
        F.xxhash64(F.concat_ws(" ", "v", "w")).alias("k_vw"),
        F.xxhash64("v").alias("k_v"),
        F.xxhash64("w").alias("k_w"))

    # 2. Se busca cada número en su tabla
    found = (keys
        .join(B(t3), "k_uvw", "left") # añade c_uvw (veces que se vio el trío) y r_uvw (puesto de w tras “on the”).
        .join(B(t2.select(F.col("k").alias("k_uv"), F.col("count").alias("c_uv"))), "k_uv", "left") # añade c_uv (veces que se vio “on the”). Es el denominador del caso 1.
        .join(B(t2.select(F.col("k").alias("k_vw"), F.col("count").alias("c_vw"),
                          F.col("rank").alias("r_vw"))), "k_vw", "left") # añade c_vw (veces que se vio “the other”), el numerador del caso 2, y r_vw (puesto de w tras “the”).
        .join(B(t1.select(F.col("k").alias("k_v"), F.col("count").alias("c_v"))), "k_v", "left") # añade c_v (veces que se vio “the”). Es el denominador del caso 2.
        .join(B(t1.select(F.col("k").alias("k_w"), F.col("count").alias("c_w"),
                          F.col("rank").alias("r_w"))), "k_w", "left")) # añade c_w (veces que se vio “other”), que se usa en el caso 3, y r_w (puesto de “other” entre todas las palabras por frecuencia).

    # 3. Se quitan los números auxiliares
    return found.drop("k_uvw", "k_uv", "k_vw", "k_v", "k_w"), N, V


def score_tokens(found: DataFrame, N: int, V: int) -> DataFrame:
    """
    Aplica Stupid Backoff a cada palabra: usa el contexto más largo que exista
    y devuelve su log-puntuación, el nivel usado y el puesto de w en ese nivel.
    """
    tri = F.col("c_uvw").isNotNull()   # ¿existe el trío?
    bi = F.col("c_vw").isNotNull()     # ¿existe la pareja?
    a = BACKOFF_ALPHA                  # 0.4

    p = (F.when(tri, F.col("c_uvw") / F.col("c_uv"))                                # caso 1
          .when(bi, a * F.col("c_vw") / F.col("c_v"))                               # caso 2
          .otherwise(a * a * (F.coalesce("c_w", F.lit(0)) + 1) / (N + V)))          # caso 3

    level = F.when(tri, 3).when(bi, 2).otherwise(1)

    rank = (F.when(tri, F.col("r_uvw"))
             .when(bi, F.col("r_vw"))
             .otherwise(F.coalesce("r_w", F.lit(V + 1))))

    return found.select("id", "pos", "w",
                        F.log(p).alias("logp"), level.alias("level"), rank.alias("rank"))

# ------------------------------

LM_COLS = ["lm_mean_logp", "lm_std_logp", "lm_p10_logp",
           "lm_frac_tri", "lm_frac_unigram",
           "lm_frac_rank1", "lm_frac_rank10", "lm_frac_rank100", "lm_mean_logrank"]


def doc_features(scored: DataFrame) -> DataFrame:
    """
    Resume las palabras de cada texto en 9 features de previsibilidad (una fila por texto).

    A partir de logp (cuánto de probable es cada palabra):
    - lm_mean_logp:    media de logp; lo previsible que es el texto en conjunto (perplejidad).
                       Se espera más alta en IA.
    - lm_std_logp:     desviación de logp; cuánto varía la previsibilidad entre palabras.
                       Se espera más alta en humanos (escritura más irregular).
    - lm_p10_logp:     percentil 10 de logp; el logp del 10 % de palabras más sorprendentes.
                       Se espera más bajo en humanos (más palabras inesperadas).

    A partir de level (qué caso del backoff se usó):
    - lm_frac_tri:     % de palabras con trío conocido (caso 1). Se espera más alto en IA.
    - lm_frac_unigram: % de palabras sin pareja conocida (caso 3), combinaciones nunca vistas.
                       Se espera más alto en humanos.

    A partir de rank (puesto de la palabra entre las continuaciones posibles, idea de GLTR):
    - lm_frac_rank1:   % de palabras que eran la favorita (puesto 1). Se espera más alto en IA.
    - lm_frac_rank10:  % de palabras en el top 10. Se espera más alto en IA.
    - lm_frac_rank100: % de palabras en el top 100. Se espera más alto en IA.
    - lm_mean_logrank: media del logaritmo del puesto. Se espera más baja en IA.

    groupBy("id") es una transformación wide (un shuffle), pero Spark agrega parcialmente
    en cada partición antes de mover nada (como reduceByKey).
    """
    frac = lambda cond: F.mean(cond.cast("double"))   # % de palabras que cumplen la condición (media de 1 y 0)

    return scored.groupBy("id").agg(
        F.mean("logp").alias("lm_mean_logp"),
        F.stddev("logp").alias("lm_std_logp"),
        F.percentile_approx("logp", 0.1).alias("lm_p10_logp"),   # aproximado: evita ordenar todas las palabras
        frac(F.col("level") == 3).alias("lm_frac_tri"),
        frac(F.col("level") == 1).alias("lm_frac_unigram"),
        frac(F.col("rank") == 1).alias("lm_frac_rank1"),
        frac(F.col("rank") <= 10).alias("lm_frac_rank10"),
        frac(F.col("rank") <= 100).alias("lm_frac_rank100"),
        F.mean(F.log("rank")).alias("lm_mean_logrank"))


def build_features(spark: SparkSession, fraction: float = None) -> None:
    """
    Calcula las features de previsibilidad de train, val y test y las guarda en features/.
    No incluye lm_fit: esos textos se usaron para construir el modelo de lenguaje (evita fuga).
    Con `fraction` (p. ej. 0.01) usa solo una muestra, para probar rápido.
    """
    docs = (spark.read.parquet(str(CLEAN_PATH))
            .filter(F.col("split").isin("train", "val", "test"))
            .select("id", "text", "label", "split", "subsource"))
    if fraction:
        docs = docs.sample(fraction=fraction, seed=SEED)

    # Cadena completa: palabra -> conteos -> puntuación -> 9 features por texto
    found, N, V = lookup_counts(spark, token_rows(docs))
    feats = doc_features(score_tokens(found, N, V))

    # Se recupera el split y se juntan las 9 columnas en un vector. Solo id + vector (+ split
    # para particionar): label se añade en load_split, si se guardase aquí quedaría duplicada
    meta = docs.select("id", "split")
    out = VectorAssembler(inputCols=LM_COLS, outputCol="features_lm").transform(meta.join(feats, "id"))

    name = "lm_sample.parquet" if fraction else "lm.parquet"
    (out.select("id", "split", "features_lm")
        .write.mode("overwrite").partitionBy("split").parquet(str(FEATURES_DIR / name)))


if __name__ == "__main__":
    import sys
    paso = sys.argv[1] if len(sys.argv) > 1 else "count"
    spark = get_spark()

    if paso == "count":
        count_ngrams(spark)
        c = spark.read.parquet(str(LM_DIR / "ngram_counts"))
        c.groupBy("n").agg(F.count("*").alias("distintos"), F.sum("count").alias("apariciones"),
                           F.sum((F.col("count") >= 2).cast("int")).alias("vistos_2_o_mas")).orderBy("n").show()
        c.filter("n = 3").orderBy(F.desc("count")).show(15, truncate=False)

    elif paso == "table":
        build_table(spark)
        t = spark.read.parquet(str(LM_DIR / "ngram_table"))
        t.groupBy("n").count().orderBy("n").show()
        t.filter("ctx = 'on the'").orderBy("rank").show(10, truncate=False)   # ejemplo

    elif paso == "load":
        t3, t2, t1, N, V = _load_tables(spark)
        print(f"N (total de palabras) = {N:,}   V (palabras distintas) = {V:,}")
        print("tríos:", t3.count(), " parejas:", t2.count(), " palabras:", t1.count())
        print("Tríos:");    t3.show(3)
        print("Parejas:");  t2.show(3)
        print("Palabras:"); t1.show(3)

    elif paso == "rows":
        doc = spark.read.parquet(str(CLEAN_PATH)).filter("split = 'val'").select("id", "text").limit(1)
        print(doc.first()["text"][:150])
        token_rows(doc).show(12, truncate=False)

    elif paso == "lookup":
        doc = spark.read.parquet(str(CLEAN_PATH)).filter("split = 'val'").limit(1)
        found, N, V = lookup_counts(spark, token_rows(doc))
        (found.orderBy("pos")
              .select("pos", "u", "v", "w",
                      "c_uvw", "c_uv", "c_vw", "c_v", "c_w",
                      "r_uvw", "r_vw", "r_w")
              .show(12, truncate=False))

    elif paso == "score":
        doc = spark.read.parquet(str(CLEAN_PATH)).filter("split = 'val'").limit(1)
        found, N, V = lookup_counts(spark, token_rows(doc))
        score_tokens(found, N, V).orderBy("pos").show(12, truncate=False)

    elif paso == "features":
        frac = float(sys.argv[2]) if len(sys.argv) > 2 else None
        build_features(spark, frac)
        f = spark.read.parquet(str(FEATURES_DIR / ("lm_sample.parquet" if frac else "lm.parquet")))
        print("textos:", f.count())
        # label vive en clean.parquet; el vector se separa de nuevo en sus 9 columnas
        f = (f.join(spark.read.parquet(str(CLEAN_PATH)).select("id", "label"), "id")
              .withColumn("v", vector_to_array("features_lm")))
        (f.groupBy("label")
          .agg(*[F.round(F.mean(F.col("v")[i]), 3).alias(c) for i, c in enumerate(LM_COLS)])
          .orderBy("label").show(truncate=False))

    spark.stop()
