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

Todo el proyecto, de la descarga de los datos a la evaluación, se ejecuta con un solo comando
desde `src/` (los módulos se importan entre sí por nombre):

~~~bash
cd src/
python3 main.py
~~~

`main.py` encadena estos pasos:

1. **Preprocesado** (`clean_data.py`): descarga el dataset de Hugging Face (solo la primera
   vez), limpia los artefactos de formato, aplica la ventana fija de 100 palabras, elimina
   duplicados y asigna los splits → `data/clean.parquet`.
2. **Features** (`features_families/`): una tabla por familia en `features/`.
3. **Entrenamiento** (`train.py`): ajuste de hiperparámetros con validación cruzada sobre una
   muestra de train y reentreno con todo train → `models/<columna>_<modelo>/`.
4. **Evaluación** (`evaluate.py`): métricas en val y test → `results/metrics.csv` y
   `results/by_subsource.csv`.
5. **Stacking** (`stacking.py`): combinación del mejor modelo de cada familia → fila `stacking`
   en `results/metrics.csv`, `results/stacking_weights.csv` y `results/stacking_ablation.csv`.

Cada paso se salta si su salida ya existe completa (Spark escribe `_SUCCESS` al terminar), así
que una ejecución interrumpida se puede relanzar sin repetir lo ya hecho. Para rehacer un paso,
basta con borrar su carpeta de salida.

### Ejecución en el CESGA

Los resultados de `results/` se obtuvieron en el CESGA, enviando `main.py` al sistema de colas:

~~~bash
# TODO: script de colas y comando de envío (p. ej. sbatch <script>)
~~~

<!-- TODO: recursos solicitados (nodos, núcleos, memoria, tiempo máximo) y cualquier
     módulo o entorno que haya que cargar antes de ejecutar -->

En un ordenador personal basta con `python3 main.py`; el tamaño de la memoria del driver se
ajusta en `get_spark` (`src/config.py`).

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
  config.py              # parámetros comunes (semilla, rutas, ventana, splits, sesión de Spark)
  clean_data.py          # carga, limpieza, ventana, duplicados y splits
  features_families/
    lexical_fam.py       # familia 1: n-gramas de palabras (con y sin stopwords)
    char_fam.py          # familia 2: n-gramas de caracteres
    style_fam.py         # familia 3: features estilométricas
    lm_fam.py            # familia 4: modelo de lenguaje de n-gramas (previsibilidad)
  load_datasets.py       # une las features de una familia con su etiqueta, por split
  models.py              # Pipelines de MLlib y rejillas de hiperparámetros
  train.py               # validación cruzada, reentreno y guardado de los modelos
  evaluate.py            # métricas en val y test, global y por fuente
  stacking.py            # meta-modelo sobre el mejor modelo de cada familia
  main.py                # ejecución completa
notebooks/
  analisis_exploratorio.ipynb   # EDA: clases, fuentes, longitudes, artefactos, limpieza
results/                 # métricas finales (CSV)
memoria/                 # memoria en formato LNCS
~~~

Carpetas generadas al ejecutar (no se suben al repositorio): `data/` (datos originales, limpios y
conteos del modelo de lenguaje), `features/` (un parquet por familia) y `models/` (modelos
entrenados).

## Reproducibilidad

- Semilla fija (`SEED = 42`) en `src/config.py`.
- La partición train / val / test se asigna con un hash del `id` (`xxhash64`), no con
  `randomSplit`: el resultado es idéntico en cualquier máquina, independientemente del número de
  particiones o núcleos.
- La eliminación de duplicados conserva siempre la copia de menor `id`, por lo que es determinista.