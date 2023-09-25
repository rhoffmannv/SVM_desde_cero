# SVM Lineal: Implementación desde cero
- En este proyecto se implementa desde cero el algoritmo SVM Lineal para realizar clasificaciones binarias y se comparan los resultados con la implementación de Scikit Learn.
- Como datos a estudiar se usa primero un *dataset* de prueba con dos cúmulos de puntos y luego se usa la base de datos Iris (plantas del genero *Iris*) para realizar la clasificación entre clases.
- El proyecto esta compuesto por dos *jupyter notebooks*:
  - **Algoritmo PCA.ipynb**: Donde se crea clase con implementación manual de Supo¿port Vector Machines Lineal.
  - **Implementacion Iris Dataset.ipynb**: Se implementa el algoritmo manual sobre los dos *datasets* mencionados y se comparan resultados con la implementación de Scikit Learn.
 
# Support Vector Machines

- SVM es un algoritmo supervisado de *machine learning* usado generalmente para problemas de clasificación.
- Este algoritmo encuentra el hiperplano que mejor separa los datos en dos clases.
- Al tomar los componentes de los datos proyectados en las direcciones con mayor variabilidad, se logra mantener gran cantidad de la información que diferencia los datos y a la vez reducir el número de componentes por datos (reducción de dimensionalidad).
- Debido a que se enfoca en mantener los datos diferenciados, permite mantener la estructura global de los datos al reducir el número de dimensiones.
- Al usar PCA se tiene la libertad de elegir cuantas direcciones principales utilizar, entre más direcciones se usan, más información se captura de los datos, pero a la vez aumenta el número de dimensiones.
