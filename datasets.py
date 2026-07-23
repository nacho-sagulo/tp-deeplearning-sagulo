"""
Pipeline de datos para `ARTransformerForecaster`, organizado en 3 capas:

- Capa 1 (funciones puras de pandas, sin PyTorch): cortes de calendario
  Train/Val/Test (`calcular_cortes_split`), elegibilidad de pozos por
  split (`elegibilidad_pozos`), armado de ventanas (`generar_ventanas`),
  fit de normalización SOLO con datos de Train (`calcular_normalizacion`,
  `calcular_normalizacion_regresores`) y filtrado de pozos por paradas
  (`calcular_paradas_por_pozo`, `filtrar_pozos_por_parada`,
  `tabla_elegibilidad_por_umbral_parada`).
- Capa 2 (`PozoWindowDataset`): transforma un DataFrame de ventanas
  (salida de Capa 1) en los tensores del modelo -- no ajusta ningún
  parámetro, solo aplica la normalización ya calculada.
- Capa 3 (`PozoDataModule`): orquesta las capas 1 y 2 en `setup()` y arma
  los `DataLoader` de train/val/test.

 En la seccion  "1.1 Manejo de ventanas y predicciones" de la mnotebook 
 transformer.ipynb se muestra en forma más detallada un ejemplo de ventanas
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

# Hiperparámetros de armado de ventanas/cortes, compartidos por todo el pipeline
# INPUT_LEN (36): cantidad de meses de historia que recibe el modelo como
# input en cada ventana (3 años) -- ver `generar_ventanas`.
# TARGET_LEN (12): cantidad de meses del horizonte a predecir por ventana
# (t+1 a t+12), el horizonte planteado en la propuesta del TP
# MIN_VENTANAS_TRAIN (12): NO es un tope de ventanas por pozo -- es el
# piso de historia mínima para calificar a Train, medido en cantidad de
# ventanas. Ejemplo con los defaults (INPUT_LEN=36, TARGET_LEN=12,
# train_start=2019-07, ver `calcular_cortes_split`): el pozo con MENOS
# historia que califica tiene su primer input en 2019-07, y
# aporta exactamente 12 ventanas (desliza el input_start mes a mes, de
# 2019-07 a 2020-06). Un pozo con fecha_min=2015-01 (más historia hacia
# atrás) aporta muchas más ventanas (66, de 2015-01 a 2020-06), porque
# `generar_ventanas` arranca desde el propio inicio de cada pozo, no
# desde train_start.
INPUT_LEN = 36
TARGET_LEN = 12
MIN_VENTANAS_TRAIN = 12


def process_dataset(nombre_archivo: str) -> pd.DataFrame:
    """
    Carga el dataset base (mismo formato de `cols_modelo` en preparacion.ipynb)
    y agrega las features derivadas que necesita el modelo. Punto único de
    entrada del pipeline previo a armar ventanas y datasets de PyTorch — acá
    se van a ir sumando las tareas de normalización pendientes.

    Agrega:
    - edad_pozo: meses transcurridos desde el inicio de producción del pozo
      (0 en su primer mes de historia).
    """
    df = pd.read_csv(nombre_archivo)

    # .asi8: ordinal entero de cada Period (freq="M") — la diferencia entre
    # dos ordinales de la misma frecuencia da directamente meses transcurridos.
    ordinal_mes = pd.PeriodIndex.from_fields(year=df["anio_prod"], month=df["mes_prod"], freq="M").asi8
    ordinal_min_pozo = pd.Series(ordinal_mes, index=df.index).groupby(df["idpozo"]).transform("min")
    df["edad_pozo"] = ordinal_mes - ordinal_min_pozo

    return df


@dataclass
class CortesSplit:
    """Cortes de calendario (pd.Period, freq='M') que delimitan Train/Val/Test."""

    fecha_max: pd.Period
    test_input_start: pd.Period
    test_target_start: pd.Period
    val_input_start: pd.Period
    val_target_start: pd.Period
    train_start: pd.Period
    train_target_max_end: pd.Period


def calcular_cortes_split(
    df: pd.DataFrame,
    input_len: int = INPUT_LEN,
    target_len: int = TARGET_LEN,
    min_ventanas_train: int = MIN_VENTANAS_TRAIN,
    col_anio: str = "anio_prod",
    col_mes: str = "mes_prod",
) -> CortesSplit:
    """
    Deriva los cortes de calendario fijos para Test, Val y Train:

    - Test:  target = últimos `target_len` meses del dataset.
    - Val:   target = los `target_len` meses inmediatamente anteriores al target de Test.
    - Train: rango cuyo target más tardío termina justo antes de que arranque
      el target de Val, con extensión suficiente para garantizar al menos
      `min_ventanas_train` ventanas deslizantes (input `input_len` + target
      `target_len`, desplazando de a 1 mes).

    Ojo: `train_start` es un piso de ELEGIBILIDAD (lo usa `elegibilidad_pozos`
    para decidir qué pozos califican para train), no un piso de generación de
    ventanas -- `generar_ventanas` usa el propio `fecha_min` de cada pozo, no
    `train_start`, así que un pozo con más historia aporta más de
    `min_ventanas_train` ventanas.

    Ejemplo (defaults: input_len=36, target_len=12, min_ventanas_train=12):
    una ventana ocupa 36+12=48 meses; cada ventana siguiente se corre 1 mes,
    así que 12 ventanas solapadas ocupan 48+11=59 meses en total (span_train).
    Si train_target_max_end=2024-05, ese rango de 59 meses (ambos extremos
    incluidos) arranca en train_start=2019-07 (2024-05 retrocediendo 58
    meses, no 59, porque el propio 2024-05 ya cuenta como uno de los 59).
    Un pozo con fecha_min=2019-07 (justo el límite) aporta exactamente 12
    ventanas; uno con fecha_min=2015-01 aporta muchas más, porque
    `generar_ventanas` usa su propio inicio, no `train_start`.
    """
    periodo = pd.PeriodIndex.from_fields(year=df[col_anio], month=df[col_mes], freq="M")
    fecha_max = periodo.max()

    test_target_start = fecha_max - (target_len - 1)
    test_input_start = test_target_start - input_len

    val_target_start = test_target_start - target_len
    val_input_start = val_target_start - input_len

    train_target_max_end = val_target_start - 1
    span_train = input_len + target_len + (min_ventanas_train - 1)
    train_start = train_target_max_end - (span_train - 1)

    return CortesSplit(
        fecha_max=fecha_max,
        test_input_start=test_input_start,
        test_target_start=test_target_start,
        val_input_start=val_input_start,
        val_target_start=val_target_start,
        train_start=train_start,
        train_target_max_end=train_target_max_end,
    )


def elegibilidad_pozos(
    df: pd.DataFrame,
    cortes: CortesSplit,
    target_len: int = TARGET_LEN,
    col_idpozo: str = "idpozo",
    col_anio: str = "anio_prod",
    col_mes: str = "mes_prod",
) -> pd.DataFrame:
    """
    Determina, por pozo, si su historia (fecha_min/fecha_max) alcanza para
    aportar ventanas a Train, Val y/o Test, dados los cortes de `cortes`.

    Nota: solo valida que la historia arranque y llegue lo suficientemente
    lejos — no valida continuidad/huecos dentro del rango.

    Devuelve un dataframe indexado por `col_idpozo` con fecha_min, fecha_max,
    y las columnas booleanas elegible_train / elegible_val / elegible_test.
    """
    periodo = pd.PeriodIndex.from_fields(year=df[col_anio], month=df[col_mes], freq="M")
    resumen_pozo = (
        df.assign(periodo=periodo)
        .groupby(col_idpozo)["periodo"]
        .agg(fecha_min="min", fecha_max="max")
    )

    resumen_pozo["elegible_train"] = (resumen_pozo["fecha_min"] <= cortes.train_start) & (
        resumen_pozo["fecha_max"] >= cortes.train_target_max_end
    )
    resumen_pozo["elegible_val"] = (resumen_pozo["fecha_min"] <= cortes.val_input_start) & (
        resumen_pozo["fecha_max"] >= cortes.val_target_start + target_len - 1
    )
    resumen_pozo["elegible_test"] = (resumen_pozo["fecha_min"] <= cortes.test_input_start) & (
        resumen_pozo["fecha_max"] >= cortes.fecha_max
    )

    return resumen_pozo


def calcular_paradas_por_pozo(
    df: pd.DataFrame,
    col_idpozo: str = "idpozo",
    col_anio: str = "anio_prod",
    col_mes: str = "mes_prod",
    col_target: str = "prod_pet",
    umbral_prod: float = 1.0,
) -> pd.DataFrame:
    """
    Calcula, para cada pozo de `df`, dos métricas de "suciedad" por meses
    de parada: la tasa `paradas_por_anio` y la racha más larga de meses
    CONSECUTIVOS de parada (`racha_maxima_parada`, calculada por tramos
    más abajo). Usa TODA la historia disponible del pozo.

    "Parada" se define acá sobre la variable objetivo: `prod_pet <=
    umbral_prod`. No se usa `tipoestado_prod`, como hace el flag `activo`
    de `PozoWindowDataset`. Es un criterio es más restrictivo: un pozo
    "activo" según su estado administrativo igual cuenta como parada si
    produce casi nada. Por eso la columna se llama `parada_prod`, no
    `activo`. Son conceptos distintos, con propósitos distintos.

    La tasa sola no alcanza. No distingue si los meses de parada están
    repartidos o concentrados en un bloque largo. Ejemplo: un pozo con 8
    años de historia y 8 meses de parada tiene `paradas_por_anio=1.0` en
    los dos casos siguientes: 1 mes por año (repartidos), u 8 meses
    seguidos (un solo bloque). En el segundo caso, esa racha larga puede
    caer entera dentro de una ventana de train/val/test y arruinarla
    igual. De ahí `racha_maxima_parada`.

    Devuelve un dataframe indexado por `col_idpozo`, con `meses_totales`,
    `meses_parada`, `paradas_por_anio` (= meses_parada / (meses_totales/12))
    y `racha_maxima_parada` (0 si el pozo nunca tuvo parada). Es un
    resultado intermedio. Lo usan `filtrar_pozos_por_parada` y
    `tabla_elegibilidad_por_umbral_parada`.
    """
    periodo = pd.PeriodIndex.from_fields(year=df[col_anio], month=df[col_mes], freq="M")
    df_ordenado = (
        df.assign(periodo=periodo, parada_prod=df[col_target] <= umbral_prod)
        .sort_values([col_idpozo, "periodo"])
    )

    resumen = df_ordenado.groupby(col_idpozo)["parada_prod"].agg(meses_totales="size", meses_parada="sum")
    resumen["paradas_por_anio"] = resumen["meses_parada"] / (resumen["meses_totales"] / 12)

    # Racha máxima: agrupo tramos consecutivos de igual valor de parada_prod
    # (cambia de tramo cuando el valor difiere del de la fila anterior DEL
    # MISMO pozo -- df_ordenado ya está ordenado por pozo+periodo, así que
    # el shift() compara con el mes inmediato anterior), y me quedo con el
    # más largo de los tramos que son de parada.
    cambio_tramo = df_ordenado["parada_prod"] != df_ordenado.groupby(col_idpozo)["parada_prod"].shift()
    id_tramo = cambio_tramo.groupby(df_ordenado[col_idpozo]).cumsum()
    tramos = (
        df_ordenado.assign(id_tramo=id_tramo)
        .groupby([col_idpozo, "id_tramo"])["parada_prod"]
        .agg(es_parada="first", longitud="size")
    )
    racha_maxima = tramos.loc[tramos["es_parada"]].groupby(level=0)["longitud"].max()
    resumen["racha_maxima_parada"] = racha_maxima.reindex(resumen.index).fillna(0).astype(int)

    return resumen


def filtrar_pozos_por_parada(
    df: pd.DataFrame,
    umbral: float,
    racha_maxima: int | None = None,
    col_idpozo: str = "idpozo",
    col_anio: str = "anio_prod",
    col_mes: str = "mes_prod",
    col_target: str = "prod_pet",
    umbral_prod: float = 1.0,
) -> pd.DataFrame:
    """
    Devuelve `df` filtrado a los pozos cuya tasa `paradas_por_anio` (ver
    `calcular_paradas_por_pozo`) es <= `umbral`, calculada sobre TODA la
    historia de cada pozo (independientemente de a qué split -- train/val/test
    -- pertenezcan después sus ventanas). `umbral=0` exige cero meses de
    parada en toda la historia del pozo.

    `racha_maxima`, si se pasa (no `None`), exige ADEMÁS que la racha más
    larga de meses consecutivos de parada del pozo no supere ese tope -- la
    tasa sola no detecta bloques largos concentrados de parada (ver
    docstring de `calcular_paradas_por_pozo`).
    """
    resumen_paradas = calcular_paradas_por_pozo(df, col_idpozo, col_anio, col_mes, col_target, umbral_prod)
    mask = resumen_paradas["paradas_por_anio"] <= umbral
    if racha_maxima is not None:
        mask &= resumen_paradas["racha_maxima_parada"] <= racha_maxima
    pozos_ok = resumen_paradas.index[mask]
    return df[df[col_idpozo].isin(pozos_ok)].copy()


def tabla_elegibilidad_por_umbral_parada(
    df: pd.DataFrame,
    cortes: CortesSplit,
    umbrales: list,
    racha_maxima: int | None = None,
    target_len: int = TARGET_LEN,
    col_idpozo: str = "idpozo",
    col_anio: str = "anio_prod",
    col_mes: str = "mes_prod",
    col_target: str = "prod_pet",
    umbral_prod: float = 1.0,
) -> pd.DataFrame:
    """
    Tabla cruzada: para cada umbral de `umbrales` (meses de parada por año,
    ver `calcular_paradas_por_pozo`), cuántos pozos filtrados por ese umbral
    (y, si se pasa `racha_maxima`, también por ese tope de racha consecutiva)
    quedan elegibles para train/val/test. Reusa `elegibilidad_pozos` (ya
    calculada una sola vez sobre TODOS los pozos, sin volver a correr el
    filtro de calendario) y la interseca con el filtro de parada de cada
    umbral.

    Devuelve un DataFrame indexado por `umbral`, con `pozos_totales`
    (cantidad de pozos que pasan ese umbral, sin restringir a elegibilidad
    de split) y `elegible_train`/`elegible_val`/`elegible_test`.
    """
    resumen_paradas = calcular_paradas_por_pozo(df, col_idpozo, col_anio, col_mes, col_target, umbral_prod)
    resumen_elegibilidad = elegibilidad_pozos(df, cortes, target_len, col_idpozo, col_anio, col_mes)

    filas = []
    for umbral in umbrales:
        mask = resumen_paradas["paradas_por_anio"] <= umbral
        if racha_maxima is not None:
            mask &= resumen_paradas["racha_maxima_parada"] <= racha_maxima
        pozos_ok = resumen_paradas.index[mask]
        elegibles_filtrados = resumen_elegibilidad.loc[resumen_elegibilidad.index.isin(pozos_ok)]
        filas.append({
            "umbral": umbral,
            "pozos_totales": len(pozos_ok),
            "elegible_train": int(elegibles_filtrados["elegible_train"].sum()),
            "elegible_val": int(elegibles_filtrados["elegible_val"].sum()),
            "elegible_test": int(elegibles_filtrados["elegible_test"].sum()),
        })
    return pd.DataFrame(filas).set_index("umbral")


def generar_ventanas(
    df: pd.DataFrame,
    cortes: CortesSplit,
    split: str,
    pozos_elegibles,
    input_len: int = INPUT_LEN,
    target_len: int = TARGET_LEN,
    col_idpozo: str = "idpozo",
    col_anio: str = "anio_prod",
    col_mes: str = "mes_prod",
) -> pd.DataFrame:
    """
    Genera las ventanas (idpozo, input_start, target_start) de un split,
    restringido a `pozos_elegibles` (ver `elegibilidad_pozos`).

    - "val"/"test": una única ventana fija por pozo, con `input_start` y
      `target_start` iguales para todos (corte de calendario fijo).
    - "train": todas las ventanas deslizantes (paso 1 mes) que caben entre
      el inicio real de la historia de cada pozo (su propio `fecha_min`,
      no `cortes.train_start`) y `cortes.train_target_max_end`. Pozos con
      más historia hacia atrás aportan más ventanas; `cortes.train_start`
      solo garantiza el mínimo (`min_ventanas_train`) para el pozo más
      nuevo que igual califica.

    Nota: no valida continuidad/huecos dentro de cada ventana (misma
    simplificación que `elegibilidad_pozos`) — un pozo con meses faltantes
    en el rango puede generar una ventana mal alineada.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split debe ser 'train', 'val' o 'test', recibido: {split!r}")

    periodo = pd.PeriodIndex.from_fields(year=df[col_anio], month=df[col_mes], freq="M")
    df_elegibles = df.assign(periodo=periodo).loc[
        df[col_idpozo].isin(pozos_elegibles), [col_idpozo, "periodo"]
    ]

    if split == "train":
        fecha_min_pozo = df_elegibles.groupby(col_idpozo)["periodo"].min()
        ultimo_input_start = cortes.train_target_max_end - input_len - target_len + 1

        filas = [
            (idpozo, input_start, input_start + input_len)
            for idpozo, fecha_min in fecha_min_pozo.items()
            for input_start in pd.period_range(fecha_min, ultimo_input_start, freq="M")
        ]
        return pd.DataFrame(filas, columns=[col_idpozo, "input_start", "target_start"])

    input_start = cortes.val_input_start if split == "val" else cortes.test_input_start
    target_start = cortes.val_target_start if split == "val" else cortes.test_target_start

    idpozos = sorted(pd.unique(df_elegibles[col_idpozo]))
    return pd.DataFrame({
        col_idpozo: idpozos,
        "input_start": input_start,
        "target_start": target_start,
    })


def calcular_normalizacion(
    df: pd.DataFrame,
    cortes: CortesSplit,
    pozos_train,
    col_idpozo: str = "idpozo",
    col_anio: str = "anio_prod",
    col_mes: str = "mes_prod",
    col_target: str = "prod_pet",
) -> pd.DataFrame:
    """
    Fit de normalización de `prod_pet`: media y desvío de log1p(prod_pet),
    por pozo, para estandarizar cada uno en su propia escala.

    Reglas anti-leakage (Val/Test no deben influir en el fit): 1) solo se
    usan filas de pozos en `pozos_train`; 2) dentro de esos pozos, solo
    filas con `periodo <= cortes.train_target_max_end` -- un pozo elegible
    para Train puede tener historia que llega hasta Val/Test, y esos
    meses futuros no deben contaminar el estadístico.

    Pozos que no están en `pozos_train` (nunca vistos en train) no tienen
    filas para este cálculo -- usan como fallback la media/std GLOBAL de
    Train. No hace falta un umbral de "mínimos datos": `elegibilidad_pozos`
    ya garantiza que todo pozo en `pozos_train` tiene la historia completa
    del rango de Train, así que no hay casos intermedios (o tiene toda la
    historia, o no tiene ninguna fila utilizable).

    Devuelve un dataframe indexado por `col_idpozo` (todos los pozos de
    `df`) con `media_log`, `std_log` y `fallback_global` (True si usó el
    fallback).
    """
    periodo = pd.PeriodIndex.from_fields(year=df[col_anio], month=df[col_mes], freq="M")
    df_fit = df.assign(periodo=periodo, log_target=np.log1p(df[col_target]))
    df_fit = df_fit.loc[
        (df_fit["periodo"] <= cortes.train_target_max_end) & df_fit[col_idpozo].isin(pozos_train),
        [col_idpozo, "log_target"],
    ]

    stats_pozo = df_fit.groupby(col_idpozo)["log_target"].agg(media_log="mean", std_log="std")

    media_global = df_fit["log_target"].mean()
    std_global = df_fit["log_target"].std()

    todos_los_pozos = pd.Index(df[col_idpozo].unique(), name=col_idpozo)
    normalizacion = stats_pozo.reindex(todos_los_pozos)
    normalizacion["fallback_global"] = normalizacion["media_log"].isna()
    normalizacion["media_log"] = normalizacion["media_log"].fillna(media_global)
    normalizacion["std_log"] = normalizacion["std_log"].fillna(std_global)

    return normalizacion


def normalizar_prod_pet(valores, media_log: float, std_log: float):
    """
    Aplica la transformación log1p + z-score a valores crudos de `prod_pet`
    (m³), usando parámetros `media_log`/`std_log` ya ajustados por
    `calcular_normalizacion`. `valores` puede ser un escalar o un array.
    """
    return (np.log1p(valores) - media_log) / std_log


def desnormalizar_prod_pet(valores_norm, media_log: float, std_log: float):
    """
    Inversa de `normalizar_prod_pet` — vuelve de z-score en escala log a
    `prod_pet` en m³. Se usa para des-normalizar las predicciones del modelo
    antes de calcular métricas en unidades reales (MAPE, R²).
    """
    return np.expm1(valores_norm * std_log + media_log)


COLUMNAS_REGRESORES_LOG1P = ["arena_bombeada_importada_tn"]
COLUMNAS_REGRESORES_DIRECTAS = [
    "profundidad_pozo",
    "longitud_rama_horizontal_m",
    "cantidad_fracturas",
    "arena_bombeada_nacional_tn",
    "agua_inyectada_m3",
    "presion_maxima_psi",
    "coordenadax",
    "coordenaday",
]


def calcular_normalizacion_regresores(
    df: pd.DataFrame,
    cortes: CortesSplit,
    pozos_train,
    col_idpozo: str = "idpozo",
    col_anio: str = "anio_prod",
    col_mes: str = "mes_prod",
    col_edad: str = "edad_pozo",
) -> pd.DataFrame:
    """
    Fit (media/std, escala POBLACIONAL — no por pozo) para los regresores
    estáticos (`COLUMNAS_REGRESORES_LOG1P` + `COLUMNAS_REGRESORES_DIRECTAS`)
    y `edad_pozo`, restringido a Train.

    A diferencia de `calcular_normalizacion` (prod_pet, por pozo), acá el
    estadístico es uno solo, compartido por todos los pozos: los regresores
    estáticos no tienen variación propia dentro de un pozo (un valor por
    pozo, no una serie), y a `edad_pozo` le conviene mantener su escala
    absoluta comparable entre pozos (es la posición en la curva de declive;
    estandarizarla por pozo destruiría esa comparación). Por el mismo
    motivo, acá NO hace falta ningún fallback para pozos fuera de train: el
    estadístico se aplica igual a cualquier pozo, visto o no.

    Dos reglas de leakage distintas según si la columna varía en el tiempo:
    - Regresores estáticos: un valor por pozo que no cambia mes a mes ->
      alcanza con restringir a pozos_train, deduplicado a 1 fila por pozo
      (si no, un pozo con más meses de historia pesaría más en el promedio,
      sesgando el estadístico hacia los pozos más viejos sin motivo).
    - edad_pozo: cambia fila a fila -> además hay que restringir a
      periodo <= train_target_max_end (mismo motivo que en
      `calcular_normalizacion`: un pozo elegible para train puede tener
      historia que cae en Val/Test).

    Devuelve un DataFrame indexado por nombre de columna, con `media`, `std`
    y `log1p` (si esa columna necesita log1p antes de estandarizar).
    """
    periodo = pd.PeriodIndex.from_fields(year=df[col_anio], month=df[col_mes], freq="M")
    df_periodo = df.assign(periodo=periodo)

    df_pozo_train = (
        df_periodo.loc[df_periodo[col_idpozo].isin(pozos_train)]
        .drop_duplicates(subset=col_idpozo)
    )
    df_fila_train = df_periodo.loc[
        df_periodo[col_idpozo].isin(pozos_train) & (df_periodo["periodo"] <= cortes.train_target_max_end)
    ]

    filas = []
    for col in COLUMNAS_REGRESORES_LOG1P:
        valores = np.log1p(df_pozo_train[col])
        filas.append((col, valores.mean(), valores.std(), True))
    for col in COLUMNAS_REGRESORES_DIRECTAS:
        valores = df_pozo_train[col]
        filas.append((col, valores.mean(), valores.std(), False))

    valores_edad = df_fila_train[col_edad]
    filas.append((col_edad, valores_edad.mean(), valores_edad.std(), False))

    return pd.DataFrame(filas, columns=["columna", "media", "std", "log1p"]).set_index("columna")


def normalizar_regresor(valores, media: float, std: float, log1p: bool = False):
    """Estandariza un regresor, con log1p opcional antes de estandarizar."""
    if log1p:
        valores = np.log1p(valores)
    return (valores - media) / std


def desnormalizar_regresor(valores_norm, media: float, std: float, log1p: bool = False):
    """Inversa de `normalizar_regresor`."""
    valores = valores_norm * std + media
    return np.expm1(valores) if log1p else valores


class PozoWindowDataset(Dataset):
    """
    Dataset de PyTorch: convierte un DataFrame de ventanas (ver
    `generar_ventanas`) en el dict de tensores de `ARTransformerForecaster`
    (x_prod_pet, x_regressors, y_regressors, y_prod_pet). El método principal
    es `__getitem__`: por cada ventana, busca los meses de input/target
    del pozo correspondiente y arma esos tensores ya normalizados.

    Es solo la capa de "transform", no ajusta ningún parámetro -- la
    normalización viene calculada de afuera, en
    `normalizacion`/`normalizacion_regresores`. Lo que es igual para todo
    el dataset (regresores estáticos, `edad_pozo`, con un estadístico
    poblacional) se normaliza una sola vez en `__init__`; lo que depende
    de cada pozo (`prod_pet`, con un estadístico distinto por pozo) se
    normaliza recién en `__getitem__`, donde ya se sabe a qué pozo
    pertenece esa ventana.

    Regresores (dimensión total 11, todos normalizados salvo `activo`):
    - Estáticos (9): completación hidráulica + coordenadas, constantes.
    - `edad_pozo` (1): dinámico pero conocido de antemano (aritmética simple).
    - `activo` (1): dinámico, SOLO en input (ver justificación abajo).

    `activo` marca si el pozo estaba en producción normal ese mes o parado
    por motivos administrativos/operativos (`tipoestado_prod` en
    `VALORES_TIPOESTADO_ACTIVO`) -- distingue una parada real de una caída
    genuina de producción, para que el modelo no confunda ambas cosas al
    aprender la curva de declive.

    Va solo en `x_regressors` (en `y_regressors` sería fuga de
    información, el estado futuro no se conoce). Como el modelo necesita
    el mismo ancho en `x_regressors` e `y_regressors`, `y_regressors`
    lleva un placeholder constante sin información real. Consecuencia:
    ayuda a limpiar la curva de declive de la historia, pero no mejora la
    predicción de paradas futuras (error irreducible por diseño).

    Metadata (nunca input): `idpozo`, `tipoestado_target` (para segmentar el
    error después) e `input_periodos`/`target_periodos` (eje temporal para
    graficar y verificar ventanas).
    """

    COLUMNAS_REGRESORES_ESTATICOS = [
        "profundidad_pozo",
        "longitud_rama_horizontal_m",
        "cantidad_fracturas",
        "arena_bombeada_nacional_tn",
        "arena_bombeada_importada_tn",
        "agua_inyectada_m3",
        "presion_maxima_psi",
        "coordenadax",
        "coordenaday",
    ]

    # "Otras Situación Activo" se suma tras verificar en test_datasets.ipynb
    # que su prod_pet se comporta como producción genuina, no como parada
    # administrativa (ver nota empírica en el docstring de la clase).
    VALORES_TIPOESTADO_ACTIVO = ["Extracción Efectiva", "Otras Situación Activo"]
    VALOR_ACTIVO_PLACEHOLDER = 1.0  # y_regressors: sin info real, solo iguala dimensión con x_regressors

    def __init__(
        self,
        df: pd.DataFrame,
        ventanas: pd.DataFrame,
        normalizacion: pd.DataFrame,
        normalizacion_regresores: pd.DataFrame,
        input_len: int = INPUT_LEN,
        target_len: int = TARGET_LEN,
        columnas_regresores_estaticos: list | None = None,
        col_idpozo: str = "idpozo",
        col_anio: str = "anio_prod",
        col_mes: str = "mes_prod",
        col_target: str = "prod_pet",
        col_edad: str = "edad_pozo",
        col_tipoestado: str = "tipoestado_prod",
    ):
        self.ventanas = ventanas.reset_index(drop=True)
        self.input_len = input_len
        self.target_len = target_len
        self.columnas_regresores_estaticos = columnas_regresores_estaticos or self.COLUMNAS_REGRESORES_ESTATICOS
        self.col_idpozo = col_idpozo
        self.col_target = col_target
        self.col_edad = col_edad
        self.col_tipoestado = col_tipoestado
        self.normalizacion = normalizacion
        self.normalizacion_regresores = normalizacion_regresores

        periodo = pd.PeriodIndex.from_fields(year=df[col_anio], month=df[col_mes], freq="M")
        df_norm = df.assign(periodo=periodo).copy()

        # edad_pozo: estadístico poblacional (ver calcular_normalizacion_regresores)
        # -> se normaliza una única vez acá, no por ventana en __getitem__.
        media_edad, std_edad, _ = normalizacion_regresores.loc[col_edad, ["media", "std", "log1p"]]
        df_norm[col_edad] = normalizar_regresor(df_norm[col_edad], media_edad, std_edad, log1p=False)

        # df indexado por (idpozo, periodo), ordenado -> lookup rápido de
        # prod_pet, edad_pozo (ya normalizado) y tipoestado_prod para
        # cualquier ventana (input y target).
        self._df_por_pozo = (
            df_norm.set_index([col_idpozo, "periodo"])[[col_target, col_edad, col_tipoestado]]
            .sort_index()
        )

        # Regresores estáticos: una fila por pozo, ya normalizados acá (mismo
        # motivo que edad_pozo: estadístico poblacional, constante por pozo).
        regresores_crudos = (
            df.drop_duplicates(subset=col_idpozo)
            .set_index(col_idpozo)[self.columnas_regresores_estaticos]
        )
        self._regresores_por_pozo = regresores_crudos.copy()
        for col in self.columnas_regresores_estaticos:
            media, std, log1p = normalizacion_regresores.loc[col, ["media", "std", "log1p"]]
            self._regresores_por_pozo[col] = normalizar_regresor(regresores_crudos[col], media, std, bool(log1p))

    def __len__(self) -> int:
        return len(self.ventanas)

    def __getitem__(self, idx: int) -> dict:
        fila = self.ventanas.iloc[idx]
        idpozo = fila[self.col_idpozo]

        input_periodos = pd.period_range(fila["input_start"], periods=self.input_len, freq="M")
        target_periodos = pd.period_range(fila["target_start"], periods=self.target_len, freq="M")

        serie_pozo = self._df_por_pozo.loc[idpozo]
        bloque_input = serie_pozo.loc[input_periodos]
        bloque_target = serie_pozo.loc[target_periodos]

        # El cast a float32 va DESPUÉS de normalizar (no antes): media_log/std_log
        # son float64 (vienen de pandas), y numpy promueve a float64 cualquier
        # resta/división entre un array float32 y un escalar float64 — si se
        # casteara antes, el resultado de normalizar_prod_pet igual terminaría
        # en float64 y rompería el matmul de nn.Linear (dtype Double vs Float).
        media_log, std_log = self.normalizacion.loc[idpozo, ["media_log", "std_log"]]
        x_prod_pet = normalizar_prod_pet(
            bloque_input[self.col_target].to_numpy(), media_log, std_log
        ).astype("float32")
        y_prod_pet = normalizar_prod_pet(
            bloque_target[self.col_target].to_numpy(), media_log, std_log
        ).astype("float32")

        # edad_pozo: regresor dinámico conocido en ambos extremos, una columna
        # extra que se concatena a los estáticos (no se "tilea", cambia mes a mes).
        edad_input = bloque_input[self.col_edad].to_numpy(dtype="float32").reshape(-1, 1)
        edad_target = bloque_target[self.col_edad].to_numpy(dtype="float32").reshape(-1, 1)

        # activo: regresor dinámico SOLO del lado del input (ver justificación
        # metodológica en el docstring de la clase — distinción zero
        # estructural/zero de muestreo). Del lado del target va un placeholder
        # constante, no el estado real, para igualar la dimensión
        # que exige ARTransformerForecaster entre x_regressor e y_regressor.
        activo_input = (
            bloque_input[self.col_tipoestado].isin(self.VALORES_TIPOESTADO_ACTIVO)
            .to_numpy(dtype="float32")
            .reshape(-1, 1)
        )
        activo_target_placeholder = np.full((self.target_len, 1), self.VALOR_ACTIVO_PLACEHOLDER, dtype="float32")

        regresores_estaticos = self._regresores_por_pozo.loc[idpozo].to_numpy(dtype="float32")
        regresores_estaticos_input = np.tile(regresores_estaticos, (self.input_len, 1))
        regresores_estaticos_target = np.tile(regresores_estaticos, (self.target_len, 1))

        x_regressors = np.hstack([regresores_estaticos_input, edad_input, activo_input])
        y_regressors = np.hstack([regresores_estaticos_target, edad_target, activo_target_placeholder])

        return {
            "x_prod_pet": torch.from_numpy(x_prod_pet).unsqueeze(-1),
            "x_regressors": torch.from_numpy(x_regressors),
            "y_regressors": torch.from_numpy(y_regressors),
            "y_prod_pet": torch.from_numpy(y_prod_pet).unsqueeze(-1),
            "idpozo": idpozo,
            # Metadata para diagnóstico posterior (nunca input del modelo):
            # estado operativo REAL del horizonte, para poder segmentar el
            # error entre meses de parada real vs. meses normales.
            "tipoestado_target": bloque_target[self.col_tipoestado].tolist(),
            # Metadata año-mes (nunca input del modelo): como string "YYYY-MM"
            # -- un pd.Period no colaciona bien en un DataLoader, un string sí,
            # y se reconstruye fácil con pd.Period(s, freq="M") si hace falta.
            # Sirve de eje temporal para graficar, y como segundo camino
            # independiente para verificar el armado de ventanas (comparar
            # contra generar_ventanas sin pasar por acá).
            "input_periodos": [str(p) for p in input_periodos],
            "target_periodos": [str(p) for p in target_periodos],
        }


class PozoDataModule(pl.LightningDataModule):
    """
    Capa 3: orquesta las Capas 1 (funciones de pandas) y 2
    (`PozoWindowDataset`) para armar los `DataLoader` de Train/Val/Test
    que consume `ARTransformerForecaster` -- es el punto de entrada único
    del pipeline de datos para el modelo.

    `setup()` hace todo el trabajo, en orden: calcula los cortes de
    calendario y la elegibilidad de cada pozo (Capa 1), ajusta la
    normalización de `prod_pet` y de los regresores UNA sola vez con
    datos de Train, genera las ventanas de cada split, y arma 5
    `PozoWindowDataset`: `train`, `val_seen`, `val_unseen`, `test_seen`,
    `test_unseen`.

    "seen"/"unseen" separa, dentro de Val y Test, los pozos que aportaron
    alguna ventana a Train de los que el modelo nunca vio -- la
    distinción de "cross-series generalization". `val_dataloader()`/
    `test_dataloader()` (los hooks que usa el Trainer) apuntan solo a los
    pozos vistos; los `_unseen` son métodos aparte, para invocar a mano
    cuando se evalúe generalización a pozos nuevos.

    `train_dataloader()` es el único con `shuffle=True`.

    Dos formas de instanciar: `PozoDataModule(ruta_datos=...)` (lee y
    procesa el CSV en `setup()`) o `PozoDataModule.from_dataframe(df, ...)`
    (usa un DataFrame ya procesado, ej. filtrado por
    `filtrar_pozos_por_parada`).
    """

    def __init__(
        self,
        ruta_datos: str | None = None,
        df: pd.DataFrame | None = None,
        batch_size: int = 32,
        input_len: int = INPUT_LEN,
        target_len: int = TARGET_LEN,
        min_ventanas_train: int = MIN_VENTANAS_TRAIN,
        num_workers: int = 0,
    ):
        super().__init__()
        if (ruta_datos is None) == (df is None):
            raise ValueError("Pasar exactamente uno de ruta_datos o df")
        self.ruta_datos = ruta_datos
        self._df_directo = df
        self.batch_size = batch_size
        self.input_len = input_len
        self.target_len = target_len
        self.min_ventanas_train = min_ventanas_train
        self.num_workers = num_workers

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **kwargs) -> "PozoDataModule":
        """
        Constructor alternativo: recibe un DataFrame ya procesado (salida de
        `process_dataset`, con `edad_pozo` ya calculado) en vez de leerlo de
        un CSV -- pensado para pipelines que filtran pozos en memoria (ver
        `filtrar_pozos_por_parada`) antes de armar el DataModule, sin pasar
        por un archivo intermedio. `setup()` usa este `df` directo tal cual,
        sin volver a llamar `process_dataset`.
        """
        return cls(df=df, **kwargs)

    def setup(self, stage: str | None = None):
        df = self._df_directo if self._df_directo is not None else process_dataset(self.ruta_datos)
        cortes = calcular_cortes_split(df, self.input_len, self.target_len, self.min_ventanas_train)
        resumen_pozo = elegibilidad_pozos(df, cortes, self.target_len)

        pozos_train = resumen_pozo.index[resumen_pozo["elegible_train"]]
        pozos_val = resumen_pozo.index[resumen_pozo["elegible_val"]]
        pozos_test = resumen_pozo.index[resumen_pozo["elegible_test"]]

        normalizacion = calcular_normalizacion(df, cortes, pozos_train)
        normalizacion_regresores = calcular_normalizacion_regresores(df, cortes, pozos_train)

        ventanas_train = generar_ventanas(df, cortes, "train", pozos_train, self.input_len, self.target_len)
        ventanas_val = generar_ventanas(df, cortes, "val", pozos_val, self.input_len, self.target_len)
        ventanas_test = generar_ventanas(df, cortes, "test", pozos_test, self.input_len, self.target_len)

        # "Vistos" = pozos que efectivamente aportaron alguna ventana a train
        # (no solo el flag elegible_train).
        pozos_vistos = set(ventanas_train["idpozo"])
        ventanas_val_seen = ventanas_val[ventanas_val["idpozo"].isin(pozos_vistos)]
        ventanas_val_unseen = ventanas_val[~ventanas_val["idpozo"].isin(pozos_vistos)]
        ventanas_test_seen = ventanas_test[ventanas_test["idpozo"].isin(pozos_vistos)]
        ventanas_test_unseen = ventanas_test[~ventanas_test["idpozo"].isin(pozos_vistos)]

        kwargs_dataset = dict(input_len=self.input_len, target_len=self.target_len)
        self.ds_train = PozoWindowDataset(df, ventanas_train, normalizacion, normalizacion_regresores, **kwargs_dataset)
        self.ds_val_seen = PozoWindowDataset(df, ventanas_val_seen, normalizacion, normalizacion_regresores, **kwargs_dataset)
        self.ds_val_unseen = PozoWindowDataset(df, ventanas_val_unseen, normalizacion, normalizacion_regresores, **kwargs_dataset)
        self.ds_test_seen = PozoWindowDataset(df, ventanas_test_seen, normalizacion, normalizacion_regresores, **kwargs_dataset)
        self.ds_test_unseen = PozoWindowDataset(df, ventanas_test_unseen, normalizacion, normalizacion_regresores, **kwargs_dataset)

        # Guardados para diagnóstico/testeo fuera de la clase.
        self.cortes = cortes
        self.resumen_pozo = resumen_pozo
        self.normalizacion = normalizacion
        self.normalizacion_regresores = normalizacion_regresores

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Hook estándar de Lightning -> pozos vistos (dirige early stopping/checkpointing)."""
        return DataLoader(self.ds_val_seen, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Hook estándar de Lightning -> pozos vistos."""
        return DataLoader(self.ds_test_seen, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader_unseen(self) -> DataLoader:
        """
        NO es un hook de Lightning (el Trainer no la llama sola) — evaluación
        de generalización a pozos nunca vistos en train, a invocar a mano
        cuando se decida medirla (ver docstring de la clase).
        """
        return DataLoader(self.ds_val_unseen, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader_unseen(self) -> DataLoader:
        """NO es un hook de Lightning — mismo motivo que `val_dataloader_unseen`."""
        return DataLoader(self.ds_test_unseen, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_input_dim(self) -> int:
        """Dimensión de x_prod_pet (canales de la historia) -> requiere setup() ya llamado."""
        return self.ds_train[0]["x_prod_pet"].shape[1]

    def get_output_dim(self) -> int:
        """Dimensión de y_prod_pet (canales del target) -> requiere setup() ya llamado."""
        return self.ds_train[0]["y_prod_pet"].shape[1]

    def get_forecast_length(self) -> int:
        """Cantidad de meses del horizonte de forecast -> requiere setup() ya llamado."""
        return self.ds_train[0]["y_prod_pet"].shape[0]

    def get_regressor_dim(self) -> int:
        """Dimensión de x_regressors/y_regressors -> requiere setup() ya llamado."""
        return self.ds_train[0]["x_regressors"].shape[1]
