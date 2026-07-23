=================================================================
TP Deep Learning - Prediccion de produccion de petroleo (Vaca Muerta)
=================================================================

Este archivo es el punto de entrada para entender el diseno de la
solucion y como ejecutarla. Para el detalle de cada componente ver el 
archivo correspondiente

-----------------------------------------------------------------
1. Como armar el entorno
-----------------------------------------------------------------

El entorno esta definido en environment.yml. Requiere
tener Conda o Miniconda instalado.

Comandos (parados en este directorio):

    conda env create -f environment.yml
    conda activate pytorch

Si el entorno ya existe y hay que actualizarlo tras un cambio en
environment.yml:

    conda env update -f environment.yml --prune

Para correr las notebooks, seleccionar el kernel "pytorch" en Jupyter o
en la extension de Jupyter de VS Code (el entorno ya incluye ipykernel).

-----------------------------------------------------------------
2. Ejecucion del modelo (camino rapido)
-----------------------------------------------------------------

Con el entorno ya armado, alcanza con abrir y correr transformer.ipynb
directamente: el repositorio ya incluye datos_modelo.csv (el archivo de
entrada del modelo, salida de preparacion.ipynb ya ejecutado), asi que
no hace falta regenerar los datos para entrenar/evaluar el modelo.

Nota: transformer.ipynb tiene un flag ENTRENAR_<...> por cada
ejecucion/configuracion (Ejecucion A, Ejecucion B, y las distintas
configuraciones de hiperparametros de la Seccion 7.6) para saltear el
entrenamiento y reusar el checkpoint ya guardado en el directorio
correspondiente -- util para iterar sobre evaluacion/graficos sin
reentrenar cada vez.

-----------------------------------------------------------------
3. Ejecucion del pipeline completo (desde los datos crudos)
-----------------------------------------------------------------

Para regenerar datos_modelo.csv desde cero (por ejemplo, si cambian los
datos crudos de SESCO en data/), correr primero preparacion.ipynb: ahi
se cargan los CSV crudos, se deciden los criterios de seleccion de
pozos/periodo de analisis, y se genera datos_modelo.csv. Recien despues
correr transformer.ipynb.

-----------------------------------------------------------------
4. Diseno general de la solucion y componentes principales
-----------------------------------------------------------------

preparacion.ipynb
    Carga los datos crudos de SESCO (produccion por pozo + datos de
    fractura), aplica los criterios de seleccion de pozos/periodo de
    analisis, y genera datos_modelo.csv -- el dataset unificado que
    consume el resto del pipeline.

datasets.py
    Pipeline de datos en 3 capas (ver su docstring para el detalle):
    - Capa 1: funciones de pandas puras -- cortes de calendario
      Train/Val/Test, elegibilidad de pozos, armado de ventanas
      input/target, ajuste de normalizacion (sin data leakage) y
      filtrado de pozos por paradas administrativas.
    - Capa 2 (PozoWindowDataset): Dataset de PyTorch, transforma
      ventanas ya armadas en los tensores que espera el modelo.
    - Capa 3 (PozoDataModule): orquesta las capas 1 y 2, arma los
      DataLoader de Train/Val/Test para PyTorch Lightning.

test_datasets.ipynb
    Suite de tests que valida y ejercita el código de datasets.py 

transformer.ipynb
    Notebook principal: define el modelo (ARTransformerForecaster,
    Transformer autoregresivo) y entrena/evalua sobre datos_modelo.csv:
    - Ejecucion A: dataset completo, incluye pozos con paradas.
    - Ejecucion B: mismo modelo, filtrando a pozos con pocas/ninguna
      parada, para dimensionar cuanto del error se explica por eso.
      Incluye ademas un ajuste de hiperparametros (Seccion 7.6) con
      varias configuraciones adicionales sobre ese mismo filtro -- ver
      la notebook para el detalle de cada una.
    Cierra con la comparacion de todas las configuraciones contra el
    criterio de aceptacion del TP, las conclusiones (Seccion 8) y las
    ideas para el trabajo final de la maestria (Seccion 9).

datos_modelo.csv
    Dataset unificado, salida de preparacion.ipynb, input de
    transformer.ipynb/datasets.py.

-----------------------------------------------------------------
5. Directorios del proyecto
-----------------------------------------------------------------

data/
    Datos crudos descargados de SESCO (pozos, produccion por pozo, datos de
    fractura) -- input de preparacion.ipynb. Los 3 CSV originales estan comprimidos en
    data/data.zip. Descomprimir ese archivo antes de correr
    preparacion.ipynb.

checkpoints/, checkpoints_b/, checkpoints_<...>/
    Checkpoints de PyTorch Lightning (.ckpt) de cada ejecucion
    entrenada -- una carpeta por ejecucion (A, B, y las que se agreguen
    mas adelante), con el nombre de archivo indicando epoca y val_loss.

lightning_logs/, lightning_logs_b/, lightning_logs_<...>/
    Logs de entrenamiento (metrics.csv por epoca/step) de PyTorch
    Lightning -- una carpeta por ejecucion, con subcarpetas version_N
    por cada corrida de esa ejecucion.
