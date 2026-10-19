import csv, time

# /pyspark
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# /src
from config import CLEAN_PATH, STYLE_PATH, MODELS_DIR, PREDICTIONS_DIR, SEED, TUNING_FRACTION, N_FOLDS, PARALLELISM
from features_style import FEATURES_ESTILO

MODELOS = ['rf', 'gbt', 'mlp']






#?  ___           _
#? (  _`\        ( )_
#? | | ) |   _ _ | ,_)   _     ___
#? | | | ) /'_` )| |   /'_`\ /',__)
#? | |_) |( (_| || |_ ( (_) )\__, \
#? (____/'`\__,_)`\__)`\___/'(____/

#? Carga de las features y muestra para el ajuste de hiperparametros


def load_style(spark: SparkSession, split: str):
    '''Features de estilo con su etiqueta, unidas por id.

    No lleva texto, asi que se puede cachear.

    Parametros:
        spark : SparkSession
        split : str
            'train', 'val' o 'test'

    Devuelve:
        DataFrame
            Con las columnas id, label, subsource y features
    '''

    etiquetas = spark.read.parquet(str(CLEAN_PATH))
    etiquetas = etiquetas.filter(F.col('split') == split)     
    etiquetas = etiquetas.select('id', 'label', 'subsource')

    features = spark.read.parquet(str(STYLE_PATH))

    return etiquetas.join(features, on='id')



def get_tuning_sample(train: DataFrame):
    '''Muestra de train para el ajuste de hiperparametros (TUNING_FRACTION), con el fold de cada fila.

    Muestra y fold salen de un hash del id, asi que no dependen del particionado
    (CrossValidator reparte por defecto con rand(seed), que si depende).
    El hash no mira la etiqueta: la muestra conserva la proporcion de clases.

    Parametros:
        train : DataFrame
            Split de train (load_style)

    Devuelve:
        DataFrame
            train filtrado, con la columna fold (0 a N_FOLDS - 1)
    '''

    # Numero de 0 a 99 por fila, con semillas distintas a las del split
    hash_muestra = F.pmod(F.xxhash64('id', F.lit(SEED + 2)), 100)
    hash_fold = F.pmod(F.xxhash64('id', F.lit(SEED + 3)), N_FOLDS)

    muestra = train.filter(hash_muestra < int(TUNING_FRACTION * 100))
    muestra = muestra.withColumn('fold', hash_fold.cast('int'))

    return muestra






#?                    _         _
#? /'\_/`\           ( )       (_ )
#? |     |   _      _| |   __   | |    _     ___
#? | (_) | /'_`\  /'_` | /'__`\ | |  /'_`\ /',__)
#? | | | |( (_) )( (_| |(  ___/ | | ( (_) )\__, \
#? (_) (_)`\___/'`\__,_)`\____)(___)`\___/'(____/

#? Estimador, rejilla de hiperparametros y forma de validar de cada modelo
#* cv  : CrossValidator con N_FOLDS folds
#* tvs : TrainValidationSplit 80/20, para los modelos caros


def get_random_forest():
    '''RandomForest y su rejilla.

    Devuelve:
        tuple : (estimador, rejilla, validacion)
    '''

    rf = RandomForestClassifier(featureSubsetStrategy='sqrt', seed=SEED)

    rejilla = (ParamGridBuilder()
               .addGrid(rf.numTrees, [100, 200])
               .addGrid(rf.maxDepth, [8, 12, 16])
               .build())

    return rf, rejilla, 'cv'



def get_gbt():
    '''GBT y su rejilla.

    Arboles poco profundos y stepSize bajo: cada arbol corrige poco y se limita el sobreajuste.

    Devuelve:
        tuple : (estimador, rejilla, validacion)
    '''

    gbt = GBTClassifier(stepSize=0.1, seed=SEED) # stepSize = LR

    rejilla = (ParamGridBuilder()
               .addGrid(gbt.maxIter, [50, 100])
               .addGrid(gbt.maxDepth, [4, 6])
               .build())

    return gbt, rejilla, 'tvs'



def get_mlp(n_features: int):
    '''MLP (con escalado) y su rejilla.

    El MLP necesita las features escaladas (la sigmoide se satura). StandardScaler es un Estimator:
    va dentro del Pipeline que se valida para que aprenda media y desviacion solo del fold de train.

    Parametros:
        n_features : int
            Neuronas de la capa de entrada

    Devuelve:
        tuple : (estimador, rejilla, validacion)
    '''

    escalador = StandardScaler(inputCol='features', outputCol='features_std', withMean=True, withStd=True)
    mlp = MultilayerPerceptronClassifier(featuresCol='features_std', seed=SEED)

    pipeline = Pipeline(stages=[escalador, mlp])

    # Capas: entrada, ocultas y salida (2 neuronas: P(humano) y P(IA))
    rejilla = (ParamGridBuilder()
               .addGrid(mlp.layers, [[n_features, 32, 2], [n_features, 64, 32, 2]])
               .addGrid(mlp.maxIter, [100, 200])
               .build())

    return pipeline, rejilla, 'tvs'



def get_model(nombre: str, n_features: int):
    '''Estimador, rejilla y forma de validar del modelo pedido.

    Parametros:
        nombre : str
            'rf', 'gbt' o 'mlp'
        n_features : int
            Tamanho del vector de features (solo lo usa el MLP)

    Devuelve:
        tuple : (estimador, rejilla, validacion)
    '''

    match nombre:

        case 'rf':
            return get_random_forest()

        case 'gbt':
            return get_gbt()

        case 'mlp':
            return get_mlp(n_features)

    raise ValueError(f'Modelo desconocido: {nombre}')






#?  _____                     _
#? (  _  )    _              ( )_
#? | (_) |   (_) _   _   ___ | ,_)   __
#? |  _  |   | |( ) ( )/',__)| |   /'__`\
#? | | | |   | || (_) |\__, \| |_ (  ___/
#? (_) (_)_  | |`\___/'(____/`\__)`\____)
#?       ( )_| |
#?       `\___/'

#? Busqueda de hiperparametros y reentreno con train completo


def formatear_parametros(parametros: dict):
    '''Convierte un ParamMap en texto ("numTrees=100, maxDepth=8").'''

    textos = []

    for param, valor in parametros.items():
        textos.append(f'{param.name}={valor}')

    return ', '.join(textos)



def tune(nombre: str, train: DataFrame):
    '''Busca los mejores hiperparametros en la muestra de ajuste y reentrena con train completo.

    Parametros:
        nombre : str
            'rf', 'gbt' o 'mlp'
        train : DataFrame
            Split de train (load_style), cacheado

    Devuelve:
        tuple : (modelo, mejores_parametros)
    '''

    estimador, rejilla, validacion = get_model(nombre, len(FEATURES_ESTILO))
    evaluador = BinaryClassificationEvaluator(metricName='areaUnderPR')

    # Se recorre combinaciones x folds veces: se cachea
    muestra = get_tuning_sample(train).cache()


    #- Busqueda en la rejilla
    if validacion == 'cv':
        ajustador = CrossValidator(estimator=estimador, estimatorParamMaps=rejilla, evaluator=evaluador,
                                   numFolds=N_FOLDS, foldCol='fold', parallelism=PARALLELISM)
        resultado = ajustador.fit(muestra)
        metricas = resultado.avgMetrics
        tipo = f'CV k={N_FOLDS}'

    else:
        ajustador = TrainValidationSplit(estimator=estimador, estimatorParamMaps=rejilla, evaluator=evaluador,
                                         trainRatio=0.8, parallelism=PARALLELISM, seed=SEED)
        resultado = ajustador.fit(muestra)
        metricas = resultado.validationMetrics
        tipo = 'TVS 80/20'

    muestra.unpersist()


    #- Resultados de cada combinacion
    print(f'[{nombre}] AUC-PR en el ajuste ({tipo}):')

    for parametros, metrica in zip(rejilla, metricas):
        print(f'  {metrica:.4f}  {formatear_parametros(parametros)}')


    #- Reentreno
    # El mejor modelo del ajuste solo ha visto la muestra: se reentrena con todo train
    mejores = rejilla[metricas.index(max(metricas))]
    modelo = estimador.fit(train, mejores)

    return modelo, mejores






#?  ___                           _             _
#? (  _`\                        ( )           ( )
#? | ( (_) _   _    _ _  _ __   _| |   _ _    _| |   _
#? | |___ ( ) ( ) /'_` )( '__)/'_` | /'_` ) /'_` | /'_`\
#? | (_, )| (_) |( (_| || |  ( (_| |( (_| |( (_| |( (_) )
#? (____/'`\___/'`\__,_)(_)  `\__,_)`\__,_)`\__,_)`\___/'

#? Modelo, importancias y predicciones de val
#* models/style_<modelo>, models/style_<modelo>_importances.csv
#* predictions/style_<modelo>_val.parquet (id, label, subsource, prob_ai) para evaluate.py


def get_importancia(par: tuple):
    '''Importancia de un par (feature, importancia). Clave para ordenar.'''

    return par[1]



def save_importances(nombre: str, modelo):
    '''Guarda en CSV la importancia de cada feature (solo RF y GBT) y muestra las 10 primeras.'''

    importancias = list(zip(FEATURES_ESTILO, modelo.featureImportances.toArray()))
    importancias.sort(key=get_importancia, reverse=True)

    ruta = MODELS_DIR / f'style_{nombre}_importances.csv'

    with open(ruta, 'w', newline='') as f:
        escritor = csv.writer(f)
        escritor.writerow(('feature', 'importance'))
        escritor.writerows(importancias)


    print(f'[{nombre}] 10 features mas importantes:')

    for feature, importancia in importancias[:10]:
        print(f'  {feature:18} {importancia:.4f}')



def save_predictions(nombre: str, modelo, val: DataFrame):
    '''Guarda las predicciones de val y muestra su AUC-PR como comprobacion rapida.'''

    ruta = str(PREDICTIONS_DIR / f'style_{nombre}_val.parquet')

    # probability = [P(humano), P(IA)]
    prob_ai = vector_to_array('probability')[1]

    predicciones = modelo.transform(val)
    predicciones = predicciones.select('id', 'label', 'subsource', prob_ai.alias('prob_ai'))
    predicciones.write.mode('overwrite').parquet(ruta)


    # Comprobacion sobre el fichero guardado (la evaluacion completa esta en evaluate.py)
    evaluador = BinaryClassificationEvaluator(rawPredictionCol='prob_ai', metricName='areaUnderPR')
    auc_pr = evaluador.evaluate(val.sparkSession.read.parquet(ruta))

    print(f'[{nombre}] AUC-PR en val: {auc_pr:.4f}')



def save_outputs(nombre: str, modelo, val: DataFrame):
    '''Guarda el modelo, sus importancias (RF y GBT) y las predicciones de val.'''

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    modelo.write().overwrite().save(str(MODELS_DIR / f'style_{nombre}'))

    # El MLP (PipelineModel) no tiene importancias
    if hasattr(modelo, 'featureImportances'):
        save_importances(nombre, modelo)

    save_predictions(nombre, modelo, val)






#?  ___                     _
#? (  _`\  _               (_ )  _
#? | |_) )(_) _ _      __   | | (_)  ___     __
#? | ,__/'| |( '_`\  /'__`\ | | | |/' _ `\ /'__`\
#? | |    | || (_) )(  ___/ | | | || ( ) |(  ___/
#? (_)    (_)| ,__/'`\____)(___)(_)(_) (_)`\____)
#?           | |
#?           (_)

#? Entrenamiento de todos los modelos


def train_style_models(spark: SparkSession, nombres: list = MODELOS):
    '''Ajusta, reentrena y guarda cada modelo de estilometria.

    Parametros:
        spark : SparkSession
        nombres : list
            Modelos a entrenar, por defecto todos (MODELOS)
    '''

    # Sin texto (~300 MB): los 3 modelos iteran sobre el, se cachea
    train = load_style(spark, 'train').cache()
    val = load_style(spark, 'val')


    for nombre in nombres:

        tiempo_inicial = time.time()

        modelo, mejores = tune(nombre, train)
        save_outputs(nombre, modelo, val)

        tiempo_total = time.time() - tiempo_inicial
        print(f'[{nombre}] mejores: {formatear_parametros(mejores)}  ({tiempo_total:.0f} s)\n')


    train.unpersist()
