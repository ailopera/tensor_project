# tensor_project
Repositorio del Trabajo de Fin de Máster "Filtrado Inteligente de Noticias basado en Deep Learning", presentado en julio de 2018.

## Estructura de directorios
- data: Modelos propios de word2vec, scripts de generación de representaciones, modelos clasificadores, script de lanzamiento de entrenamientos/validaciones + test.
  - executionStats: Ficheros csv de los experimentos realizados.
  - fnc-1-original: Datasets base del desafío, particiones creadas y script de tratamiento y limpieza de datos.
  - old: versiones anteriores del código.
  - plots: gráficas generadas.
- tutorials: Código de ejemplo para el aprendizaje de tecnologías
- R workspace: Scripts de obtención de estadísticas de los distintos particionados utilizados.
- webapp: Api y middleware de acceso a modelos del servicio web. 

## Entorno
Este repositorio usa virtualenv. Para instalar virtualenv y otra serie de dependencias asociadas, ejecutar:
```sudo pip install -r requirements.txt```

- Para crear el entorno, seguir estas instrucciones: https://www.tensorflow.org/install/install_linux
- Para arrancarlo:
```source ./bin/activate```
- Para pararlo:
```deactivate```
