# ¿Humano o máquina? Detección distribuida de texto generado por IA

Práctica 1 de Aprendizaje Automático a Gran Escala (UDC, 2026/27).
Clasificación de textos humanos frente a generados por IA con **Spark MLlib**, sobre el dataset
[artem9k/ai-text-detection-pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile)
(1,39 M de textos, 1,9 GB).

**Autores:** Alicia Losada Sánchez, Carlos Francisco Orejuela Lluncor, Hugo Varela Sánchez.

## Requisitos

- Python 3.10 o superior
- Java 17 o 21 (`java -version`)
- **No tener la variable `SPARK_HOME` apuntando a otra instalación de Spark.** PySpark usa su
  propia JVM; si `SPARK_HOME` apunta a otra versión, aparecen errores de incompatibilidad.

## Instalación

~~~bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
~~~

## Ejecución

### 1. Preprocesado

~~~bash
cd src/
python3 clean_data.py
~~~

Descarga el dataset de Hugging Face (solo la primera vez) y genera `data/clean.parquet`:
limpieza de artefactos de formato, ventana fija de 100 palabras, eliminación de duplicados
y partición en train / val / test.

### 2. Pipeline completo

*(Pendiente: `spark-submit src/main.py` ejecutará todo de principio a fin.)*

## Datos generados: `data/clean.parquet`

| Columna | Descripción |
|---|---|
| `id` | Identificador original del texto |
| `text` | Texto limpio, exactamente 100 palabras |
| `label` | 1.0 = IA, 0.0 = humano |
| `source` | Clase original (`human` / `ai`) |
| `subsource` | Fuente de origen inferida (solo para análisis, **nunca como feature**) |
| `n_words_orig` | Número de palabras del texto limpio antes de recortar |
| `split` | `train`, `val`, `test` o `lm_fit` (reservado para el modelo de n-gramas) |

El parquet está particionado por `split`: leer solo una partición evita cargar el resto.

## Estructura del repositorio

~~~
src/
  config.py              # parámetros comunes (semilla, rutas, ventana, splits)
  data.py                # carga, limpieza, ventana, duplicados y splits
  features_lexical.py    # TF-IDF de palabras y caracteres
  features_style.py      # features estilométricas
  features_lm.py         # modelo de lenguaje de n-gramas (pseudo-perplejidad)
  models.py              # modelos de MLlib y ajuste de hiperparámetros
  stacking.py            # combinación de modelos
  evaluate.py            # métricas y experimentos
  main.py                # ejecución completa
notebooks/
  analisis_exploratorio.ipynb   # EDA: clases, fuentes, longitudes, artefactos, limpieza
memoria/                 # memoria en formato LNCS
~~~

## Reproducibilidad

- Semilla fija (`SEED = 42`) en `src/config.py`.
- La partición train / val / test se asigna con un hash del `id` (`xxhash64`), no con
  `randomSplit`: el resultado es idéntico en cualquier máquina, independientemente del número de
  particiones o núcleos.
- La eliminación de duplicados conserva siempre la copia de menor `id`, por lo que es determinista.