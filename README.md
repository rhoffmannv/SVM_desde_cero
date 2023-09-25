# SVM Lineal: Implementación desde cero
- En este proyecto se implementa desde cero el algoritmo SVM Lineal para realizar clasificaciones binarias y se comparan los resultados con la implementación de Scikit Learn.
- Como datos a estudiar se usa primero un *dataset* de prueba con dos cúmulos de puntos y luego se usa la base de datos Iris (plantas del genero *Iris*) para realizar la clasificación entre clases.
- El proyecto esta compuesto por dos *jupyter notebooks*:
  - **Algoritmo PCA.ipynb**: Donde se crea clase con implementación manual de Supo¿port Vector Machines Lineal.
  - **Implementacion Iris Dataset.ipynb**: Se implementa el algoritmo manual sobre los dos *datasets* mencionados y se comparan resultados con la implementación de Scikit Learn.
 
# Support Vector Machines

- SVM es un algoritmo supervisado de *machine learning* usado generalmente para problemas de clasificación.
- Este algoritmo funciona encontrando el hiperplano que mejor separa los datos en dos clases, separando el espacio de los datos en dos.
- En su versión original el hiperplano define una separación lineal de los datos, pero se puede usar el truco del Kernel para encontrar separaciones no lineales de los datos.
- Originalmente se utiliza para problemas de clasificación binario, pero se puede usar para problemas multiclase, separándolo en una serie de problemas binarios (con métodos *One vs One* o *One vs All* por ejemplo).

<p align="center"><img src="images/svm_esquema.png"></img></p>
  
> En este proyecto se implementa el algoritmo lineal, usando el método del gradiente descendiente para calcular el hiperplano óptimo y se usa para clasificación binaria entre dos clases.

# Detalles del Proyecto

A grandes rasgos el proyecto se divide en:

- Creación de clase SVM
  - Definición de constructor
  - Definición de método *fit*
  - Definición de método *transform*
- Implementación en *dataset* de cúmulos
  - Creación de *dataset* de prueba
  - Aplicación de algoritmo manual
  - Aplicación de algoritmo de Scikit-Learn
  - Comparación gráfica de resultados.
- Implementación en *dataset Iris*
  - Importacion de datos
  - Aplicación de algoritmo manual
  - Aplicación de algoritmo de Scikit-Learn
  - Comparación gráfica de resultados.
