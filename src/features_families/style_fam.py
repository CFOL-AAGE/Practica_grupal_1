from operator import add

# /pyspark
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StopWordsRemover

# /src
from config import CLEAN_PATH, STYLE_PATH






#?  ___                        _                  _
#? (  _`\                     ( )_               ( )_
#? | ( (_)   _     ___    ___ | ,_)   _ _   ___  | ,_)   __    ___
#? | |  _  /'_`\ /' _ `\/',__)| |   /'_` )/' _ `\| |   /'__`\/',__)
#? | (_( )( (_) )| ( ) |\__, \| |_ ( (_| || ( ) || |_ (  ___/\__, \
#? (____/'`\___/'(_) (_)(____/`\__)`\__,_)(_) (_)`\__)`\____)(____/

#? Constantes de las features


# Signos de puntuacion (nombre : regex)
PUNTUACION = {

    # Pausas
    'coma' : ',',
    'punto_y_coma' : ';',
    'dos_puntos' : ':',

    # Entonacion
    'exclamacion' : '!',
    'interrogacion' : r'\?',

    # Otros
    'comillas' : '"',
    'parentesis' : r'\(',
    'guion' : r'[-\u2013\u2014]',       # guion, raya corta y raya larga
}


# Fin de frase: punto, exclamacion o interrogacion, quiza seguido de comillas o parentesis pero no obligatorio. 
FIN_FRASE = r'''[.!?]["')]?'''

# Histograma de longitudes de palabra: de 1 a 9 letras y 10 o mas
MAX_LONG_PALABRA = 10


# Nombre de cada feature en el orden en el que entra al vector.
FEATURES_ESTILO = [

    # Riqueza lexica
    'ttr',              # type-token ratio: palabras distintas / palabras
    'ratio_hapax',      # palabras que aparecen una sola vez / palabras

    # Longitud de palabra, en letras
    'long_palabra_media',
    'long_palabra_std',
    'long_palabra_1',
    'long_palabra_2',
    'long_palabra_3',
    'long_palabra_4',
    'long_palabra_5',
    'long_palabra_6',
    'long_palabra_7',
    'long_palabra_8',
    'long_palabra_9',
    'long_palabra_10_o_mas',

    # Frases, longitud en palabras
    'n_frases',
    'long_frase_media',
    'long_frase_std',
    'long_frase_cv',    # coeficiente de variacion (std / media)

    # Puntuacion 
    'punt_coma',
    'punt_punto_y_coma',
    'punt_dos_puntos',
    'punt_exclamacion',
    'punt_interrogacion',
    'punt_comillas',
    'punt_parentesis',
    'punt_guion',

    # Forma
    'ratio_mayusculas',
    'ratio_digitos',
    'n_contracciones',

    # Repeticion
    'rep_bigramas',     # 1 - bigramas distintos / bigramas 
    'rep_trigramas',    # 1 - trigramas distintos / trigramas 

    # Stopwords
    'ratio_stopwords',
]



def get_stopwords():
    '''Stopwords en ingles que trae Spark MLlib (181).'''

    return StopWordsRemover.loadDefaultStopWords('english')






#?  _____                  _
#? (  _  )              _ (_ )  _
#? | (_) | _   _       (_) | | (_)   _ _  _ __   __    ___
#? |  _  |( ) ( )(`\/')| | | | | | /'_` )( '__)/'__`\/',__)
#? | | | || (_) | >  < | | | | | |( (_| || |  (  ___/\__, \
#? (_) (_)`\___/'(_/\_)(_)(___)(_)`\__,_)(_)  `\____)(____/

#? Operaciones sobre columnas que se repiten en varias features


def ratio(num, den):
    '''num / den, o 0 si den es 0.'''

    return F.coalesce(F.try_divide(num, den), F.lit(0.0))



def contar(array, condicion):
    '''Numero de elementos de una columna array que cumplen la condicion.'''

    return F.size(F.filter(array, condicion))



def contar_iguales(array, valor):
    '''Numero de elementos de una columna array iguales a valor: totales menos los que quedan al quitar valor.'''

    return F.size(array) - F.size(F.array_remove(array, valor))



def media(array):
    '''Media de los elementos de una columna array.'''

    suma = F.aggregate(array, F.lit(0.0), add)

    return ratio(suma, F.size(array))



def cuadrado_diferencia(x, m):
    '''Paso de F.zip_with: (elemento - media)^2.'''

    return (x - m) ** 2



def desviacion(array, media_array):
    '''Desviacion tipica de los elementos de una columna array.

    La media se repite en un array del mismo tamanho para emparejarla con cada elemento (zip_with).
    '''

    medias = F.array_repeat(media_array, F.size(array))
    cuadrados = F.zip_with(array, medias, cuadrado_diferencia)

    suma_cuadrados = F.aggregate(cuadrados, F.lit(0.0), add)

    return F.sqrt(ratio(suma_cuadrados, F.size(array)))



def fraccion_repetidos(array):
    '''Fraccion de elementos repetidos de una columna array: 1 - distintos / totales.'''

    repetidos = F.size(array) - F.size(F.array_distinct(array))

    return ratio(repetidos, F.size(array))






#?  _____
#? (  _  )
#? | (_) | _ __  _ __   _ _  _   _   ___
#? |  _  |( '__)( '__)/'_` )( ) ( )/',__)
#? | | | || |   | |  ( (_| || (_) |\__, \
#? (_) (_)(_)   (_)  `\__,_)`\__, |(____/
#?                          ( )_| |
#?                          `\___/'

#? Columnas auxiliares a partir de las que se calculan las features


def limpiar_palabra(token):
    '''Deja solo las letras y el apostrofo interno de un token, en minusculas ("Don't," -> "don't").'''

    solo_letras = F.regexp_replace(token, r"[^A-Za-z']", '')
    minusculas = F.lower(solo_letras)

    return F.regexp_replace(minusculas, r"^'+|'+$", '')



def no_vacia(palabra):
    '''True si la palabra tiene algun caracter.'''

    return palabra != ''



def contar_letras(palabra):
    '''Numero de letras de una palabra (el apostrofo no cuenta).'''

    return F.length(F.regexp_replace(palabra, "'", ''))



def es_frase_completa(frase, i):
    '''True salvo para la ultima frase si el texto no acaba en fin de frase (la ha cortado la ventana).'''

    acaba_bien = F.col('text').rlike(FIN_FRASE + '$')
    no_es_la_ultima = i < F.size('frases_todas') - 1

    return acaba_bien | no_es_la_ultima



def contar_palabras(frase):
    '''Numero de palabras de una frase.'''

    return F.size(F.split(frase, ' '))



def prepare_arrays(df: DataFrame):
    '''Crea las columnas array que usan el resto de funciones.

    Parametros:
        df : DataFrame
            Con la columna text (100 palabras separadas por un espacio)

    Devuelve:
        DataFrame
            df con las columnas:
                palabras      : palabras limpias (sin puntuacion ni numeros)
                long_palabras : numero de letras de cada palabra
                frases_todas  : el texto partido por los fines de frase
                frases        : solo las frases completas
                long_frases   : numero de palabras de cada frase
    '''

    texto = F.col('text')


    #- Palabras
    tokens = F.split(texto, ' ')
    palabras = F.filter(F.transform(tokens, limpiar_palabra), no_vacia)

    df = df.withColumn('palabras', palabras)
    df = df.withColumn('long_palabras', F.transform('palabras', contar_letras))


    #- Frases
    # La ventana de 100 palabras corta la ultima frase: si el texto no acaba en fin de frase, se descarta
    df = df.withColumn('frases_todas', F.split(texto, rf'(?<={FIN_FRASE})\s+'))
    df = df.withColumn('frases', F.filter('frases_todas', es_frase_completa))
    df = df.withColumn('long_frases', F.transform('frases', contar_palabras))


    return df






#?  _
#? ( )                  _
#? | |       __        (_)   ___    _
#? | |  _  /'__`\(`\/')| | /'___) /'_`\
#? | |_( )(  ___/ >  < | |( (___ ( (_) )
#? (____/'`\____)(_/\_)(_)`\____)`\___/'

#? Riqueza lexica


def contar_apariciones(palabra):
    '''Veces que aparece palabra en la columna palabras del mismo texto.'''

    return contar_iguales(F.col('palabras'), palabra)



def get_lexical_features(df: DataFrame):
    '''Riqueza del vocabulario.

    Parametros:
        df : DataFrame
            Con la columna palabras (prepare_arrays)

    Devuelve:
        DataFrame
            df con las columnas:
                ttr         : palabras distintas / palabras
                ratio_hapax : palabras que aparecen una sola vez / palabras
    '''

    palabras = F.col('palabras')
    n_palabras = F.size(palabras)

    # Veces que aparece cada palabra distinta
    apariciones = F.transform(F.array_distinct(palabras), contar_apariciones)


    df = df.withColumns({
        'ttr' : ratio(F.size(F.array_distinct(palabras)), n_palabras),
        'ratio_hapax' : ratio(contar_iguales(apariciones, 1), n_palabras),
    })

    return df






#?  ___          _           _
#? (  _`\       (_ )        ( )
#? | |_) )  _ _  | |    _ _ | |_    _ __   _ _   ___
#? | ,__/'/'_` ) | |  /'_` )| '_`\ ( '__)/'_` )/',__)
#? | |   ( (_| | | | ( (_| || |_) )| |  ( (_| |\__, \
#? (_)   `\__,_)(___)`\__,_)(_,__/'(_)  `\__,_)(____/

#? Longitud de las palabras


def es_palabra_larga(long_palabra):
    '''True si la palabra tiene MAX_LONG_PALABRA letras o mas (ultima barra del histograma).'''

    return long_palabra >= MAX_LONG_PALABRA



def get_word_length_features(df: DataFrame):
    '''Longitud de las palabras: media, desviacion e histograma.

    Parametros:
        df : DataFrame
            Con las columnas palabras y long_palabras (prepare_arrays)

    Devuelve:
        DataFrame
            df con las columnas:
                long_palabra_media, long_palabra_std
                long_palabra_k    : % de palabras con k letras (k de 1 a 9)
                long_palabra_10_o_mas
    '''

    long_palabras = F.col('long_palabras')
    n_palabras = F.size('palabras')
    media_long = media(long_palabras)


    nuevas = {
        'long_palabra_media' : media_long,
        'long_palabra_std' : desviacion(long_palabras, media_long),
    }

    # Histograma
    for k in range(1, MAX_LONG_PALABRA):
        nuevas[f'long_palabra_{k}'] = ratio(contar_iguales(long_palabras, k), n_palabras)

    nuevas[f'long_palabra_{MAX_LONG_PALABRA}_o_mas'] = ratio(contar(long_palabras, es_palabra_larga), n_palabras)


    return df.withColumns(nuevas)






#?  ___
#? (  _`\
#? | (_(_)_ __   _ _   ___    __    ___
#? |  _) ( '__)/'_` )/',__) /'__`\/',__)
#? | |   | |  ( (_| |\__, \(  ___/\__, \
#? (_)   (_)  `\__,_)(____/`\____)(____/

#? Longitud de las frases


def get_sentence_features(df: DataFrame):
    '''Numero de frases y su longitud.

    Parametros:
        df : DataFrame
            Con las columnas frases y long_frases (prepare_arrays)

    Devuelve:
        DataFrame
            df con las columnas:
                n_frases         : frases completas
                long_frase_media : palabras por frase
                long_frase_std   : desviacion de las palabras por frase
                long_frase_cv    : coeficiente de variacion (std / media)
    '''

    long_frases = F.col('long_frases')
    media_long = media(long_frases)
    std_long = desviacion(long_frases, media_long)


    df = df.withColumns({
        'n_frases' : F.size('frases'),
        'long_frase_media' : media_long,
        'long_frase_std' : std_long,
        'long_frase_cv' : ratio(std_long, media_long),
    })

    return df






#?  ___                  _
#? (  _`\               ( )_                       _
#? | |_) ) _   _   ___  | ,_) _   _    _ _    ___ (_)   _     ___
#? | ,__/'( ) ( )/' _ `\| |  ( ) ( ) /'_` ) /'___)| | /'_`\ /' _ `\
#? | |    | (_) || ( ) || |_ | (_) |( (_| |( (___ | |( (_) )| ( ) |
#? (_)    `\___/'(_) (_)`\__)`\___/'`\__,_)`\____)(_)`\___/'(_) (_)

#? Signos de puntuacion


def get_punctuation_features(df: DataFrame):
    '''Veces que aparece cada signo de PUNTUACION.

    Parametros:
        df : DataFrame
            Con la columna text

    Devuelve:
        DataFrame
            df con una columna punt_<nombre> por cada signo
    '''

    texto = F.col('text')
    nuevas = {}

    for nombre, regex in PUNTUACION.items():
        nuevas[f'punt_{nombre}'] = F.regexp_count(texto, F.lit(regex))


    return df.withColumns(nuevas)






#?  ___
#? (  _`\
#? | (_(_)_    _ __   ___ ___     _ _
#? |  _)/'_`\ ( '__)/' _ ` _ `\ /'_` )
#? | | ( (_) )| |   | ( ) ( ) |( (_| |
#? (_) `\___/'(_)   (_) (_) (_)`\__,_)

#? Mayusculas, digitos y contracciones


def es_contraccion(palabra):
    '''True si la palabra acaba en n't, 's, 're, 've, 'll, 'd o 'm.'''

    return palabra.rlike(r"(n't|'s|'re|'ve|'ll|'d|'m)$")



def get_shape_features(df: DataFrame):
    '''Forma de los caracteres y las palabras.

    Parametros:
        df : DataFrame
            Con las columnas text y palabras (prepare_arrays)

    Devuelve:
        DataFrame
            df con las columnas:
                ratio_mayusculas : letras mayusculas / letras
                ratio_digitos    : digitos / caracteres
                n_contracciones  : palabras acabadas en n't, 's, 're, 've, 'll, 'd o 'm
    '''

    texto = F.col('text')

    n_letras = F.regexp_count(texto, F.lit('[A-Za-z]'))
    n_mayusculas = F.regexp_count(texto, F.lit('[A-Z]'))
    n_digitos = F.regexp_count(texto, F.lit('[0-9]'))
    n_contracciones = contar('palabras', es_contraccion)


    df = df.withColumns({
        'ratio_mayusculas' : ratio(n_mayusculas, n_letras),
        'ratio_digitos' : ratio(n_digitos, F.length(texto)),
        'n_contracciones' : n_contracciones,
    })

    return df






#?  ___                         _
#? |  _`\                      ( )_  _         _
#? | (_) )   __   _ _      __  | ,_)(_)   ___ (_)   _     ___
#? | ,  /  /'__`\( '_`\  /'__`\| |  | | /'___)| | /'_`\ /' _ `\
#? | |\ \ (  ___/| (_) )(  ___/| |_ | |( (___ | |( (_) )| ( ) |
#? (_) (_)`\____)| ,__/'`\____)`\__)(_)`\____)(_)`\___/'(_) (_)
#?               | |
#?               (_)

#? Repeticion de frases hechas


def desde_la_segunda(palabra, i):
    '''True para todas las palabras menos la primera.'''

    return i >= 1



def desde_la_tercera(palabra, i):
    '''True para todas las palabras menos las dos primeras.'''

    return i >= 2



def formar_bigrama(palabra, i):
    '''Bigrama que acaba en palabra: palabras[i] + palabra.'''

    palabras = F.col('palabras')

    return F.concat_ws(' ', F.get(palabras, i), palabra)



def formar_trigrama(palabra, i):
    '''Trigrama que acaba en palabra: palabras[i] + palabras[i + 1] + palabra.'''

    palabras = F.col('palabras')

    return F.concat_ws(' ', F.get(palabras, i), F.get(palabras, i + 1), palabra)



def get_repetition_features(df: DataFrame):
    '''Fraccion de bigramas y trigramas repetidos.

    Parametros:
        df : DataFrame
            Con la columna palabras (prepare_arrays)

    Devuelve:
        DataFrame
            df con las columnas:
                rep_bigramas  : 1 - bigramas distintos / bigramas
                rep_trigramas : 1 - trigramas distintos / trigramas
    '''

    palabras = F.col('palabras')

    # Quitando las primeras palabras, el elemento i es palabras[i + 1] (o palabras[i + 2]):
    # se le pegan delante las palabras anteriores para formar el n-grama
    bigramas = F.transform(F.filter(palabras, desde_la_segunda), formar_bigrama)

    trigramas = F.transform(F.filter(palabras, desde_la_tercera), formar_trigrama)


    df = df.withColumns({
        'rep_bigramas' : fraccion_repetidos(bigramas),
        'rep_trigramas' : fraccion_repetidos(trigramas),
    })

    return df






#?  ___    _                                             _
#? (  _`\ ( )_                                          ( )
#? | (_(_)| ,_)   _    _ _    _   _   _    _    _ __   _| |  ___
#? `\__ \ | |   /'_`\ ( '_`\ ( ) ( ) ( ) /'_`\ ( '__)/'_` |/',__)
#? ( )_) || |_ ( (_) )| (_) )| \_/ \_/ |( (_) )| |  ( (_| |\__, \
#? `\____)`\__)`\___/'| ,__/'`\___x___/'`\___/'(_)  `\__,_)(____/
#?                    | |
#?                    (_)

#? Stopwords (palabras funcionales)


def get_stopword_features(df: DataFrame, stopwords: list):
    '''Proporcion de stopwords.

    Usa StopWordsRemover de MLlib: quita las stopwords y la diferencia de tamanho es cuantas habia.
    Es un Transformer, no aprende nada de los datos, asi que no hay fuga de informacion.

    Parametros:
        df : DataFrame
            Con la columna palabras (prepare_arrays)
        stopwords : list
            Lista de get_stopwords()

    Devuelve:
        DataFrame
            df con las columnas:
                ratio_stopwords : stopwords / palabras
    '''

    remover = StopWordsRemover(inputCol='palabras', outputCol='palabras_sin_stopwords', stopWords=stopwords)

    df = remover.transform(df)


    n_stopwords = F.size('palabras') - F.size('palabras_sin_stopwords')
    df = df.withColumn('ratio_stopwords', ratio(n_stopwords, F.size('palabras')))

    return df






#?  ___                     _
#? (  _`\  _               (_ )  _
#? | |_) )(_) _ _      __   | | (_)  ___     __
#? | ,__/'| |( '_`\  /'__`\ | | | |/' _ `\ /'__`\
#? | |    | || (_) )(  ___/ | | | || ( ) |(  ___/
#? (_)    (_)| ,__/'`\____)(___)(_)(_) (_)`\____)
#?           | |
#?           (_)

#? Calculo y escritura de las features


def get_style_features(df: DataFrame, stopwords: list):
    '''Calcula todas las features de estilo.

    Parametros:
        df : DataFrame
            Con la columna text
        stopwords : list
            Lista de get_stopwords()

    Devuelve:
        DataFrame
            Las columnas de df y las de FEATURES_ESTILO
    '''

    columnas_iniciales = df.columns

    df = prepare_arrays(df)

    df = get_lexical_features(df)
    df = get_word_length_features(df)
    df = get_sentence_features(df)
    df = get_punctuation_features(df)
    df = get_shape_features(df)
    df = get_repetition_features(df)
    df = get_stopword_features(df, stopwords)

    # Quitamos las columnas auxiliares
    return df.select(*columnas_iniciales, *FEATURES_ESTILO)



def build_style_features(spark: SparkSession):
    '''Genera features/style.parquet con las columnas id y features (vector).

    lm_fit se salta: no lo usa ningun clasificador.
    '''

    stopwords = get_stopwords()

    df = spark.read.parquet(str(CLEAN_PATH))
    df = df.filter(F.col('split') != 'lm_fit')      
    df = df.select('id', 'text')

    df = get_style_features(df, stopwords)


    assembler = VectorAssembler(inputCols=FEATURES_ESTILO, outputCol='features')
    df = assembler.transform(df)

    df.select('id', 'features').write.mode('overwrite').parquet(str(STYLE_PATH))