## Data
Este directorio alberga los datasets de entrada del modelo y los scripts desarrollados para la creación y experimentación con las representaciones de datos. 

A continuación se listan los distintos scripts:

- Scripts de tratamiento de datos:
  - aggregateData.py: Realiza un agregado de los ficheros de csv de partida proporcionados por el desafío de Fake News.
  - cleanData.py: Script para realizar la limpieza de datasets de entrenamiento y test, de acuerdo a lo descrito en la subsección "Tratamiento de los datos de entrada".

- Scripts de ejecución de validaciones:
  - validateModel.py: Realiza la validación / test de la experimentación de modelos de representación.
  - validateClasiffier.py: Realiza la validación / test de la experimentación con modelos MLP.

- Programas de generación de representaciones y llamada a modelos:
  - BOWModel2.py: Genera la representación basada en bolsa de palabras.
  - vectorAverage.py: Representación de vector de medias.
  
- Clasificadores:
  - randomForestClassifier.py: Clasificador de Random Forest.
  - textModelClassifier.py: Clasificador multicapa con configuración fija, utilizado en la experimentación de modelos de representación.
  - textModelClassifierParametrized.py: Clasificador multicapa con configuración parametrizada, utilizado en la experimentación del modelo MLP.
- Modelos de word2vec entrenados con los datos de entrenamiento:
  - Modelo utilizado durante la experimentación: 
     - 300features_15minwords_10contextALL: Vectores de 300 dimensiones, frecuencia mínima de término de 15, tamaño de ventana 10.
  - Modelos descartados:
     - 300features_10minwords_10contextALL
     - 300features_15minwords_20contextALL
     - 300features_15minwords_35contextALL

- Programas varios:
  - loadWord2VecModel: Carga en memoria el modelo de word2vec pasado como parámetro y realiza una serie de operaciones sobre él. A partir de este fichero se han obtenido las capturas de pantalla del subapartado de "Modelos basados en Word2Vec"
  - utils.py: Se encarga de generar las curvas ROC de la configuración solicitada.
  
- Experimentación adicional (no incluida en la memoria final):
  - recurrentClassifier.py: Modelo de red recurrente con células LSTM, con número de capas configurable.
  - embeddingVector.py : Representación de vector de vectores (representantdo las distintas frases del sistema). Con algunos errores y sin ejecución óptima.
  - clusterizeWord2Vec.py: Representación basada en media de centroides. No se consiguió una ejecución óptima, por lo que no se incluyó en la experimentación de modelos de representación. Versión 2 del programa: clusterizeWord2Vec2.py
  - kFoldValidation.py: Ejecuta una validación KFold. Se trata de la validación que iba a hacer en un principio.
  - old/oldClasiffierExecutions.py: Configuraciones descartadas de la experimentación de modelos basados en MLP.
