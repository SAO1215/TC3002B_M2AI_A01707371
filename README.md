# TC3002B_M2AI_A01707371

Desarrollo de aplicaciones avanzadas de ciencias computacionales

Olivia Araceli Morales Quezada - A01707371

## Tom OR/AND Jerry

### Dataset

Este conjunto de datos contiene más de 5.000 imágenes (exactamente 5478 imágenes) extraídas de algunos de los vídeos de programas de Tom & Jerry, que están disponibles en línea.
Los videos descargados se convierten en imágenes con 1 cuadro por segundo (1 FPS).

Las imágenes etiquetadas se separan en 4 carpetas diferentes como se indica.

* Carpeta - tom_and_jerry
* SubCarpeta tom: contiene imágenes solo con 'tom'
* Subcarpeta jerry: contiene imágenes solo con 'jerry'
* Subcarpeta tom_jerry_1: contiene imágenes con 'tom' y 'jerry'
* Subcarpeta tom_jerry_0: contiene imágenes sin ambos caracteres

El archivo ground_truth.csv contiene datos etiquetados en cada archivo de imagen.

El dataset fue creado por BALA BASKAR y obtenido de Kaggle, [Tom and Jerry Image classification](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification), es una colección de más de 5.000 imágenes con datos etiquetados del programa de dibujos animados de Tom y Jerry.

### Preprocesamiento

El preprocesamiento de las imágenes es un paso crucial para preparar nuestros datos para el entrenamiento del modelo. Este proceso incluye:

* **Redimensionamiento de Imágenes:** Todas las imágenes se redimensionan a 150x150 píxeles.
* **Normalización:** Los valores de píxeles se escalan para estar en el rango [0, 1] para mejorar la convergencia durante el entrenamiento.
* **División de Datos:** El conjunto de datos se divide en 70% para entrenamiento, 20% para validación y 10% para prueba utilizando la biblioteca `splitfolders`.

### Carga de Datos con TensorFlow y Keras

Se usa `tf.keras.preprocessing.image.ImageDataGenerator` para cargar y transformar las imágenes en lotes, facilitando el manejo durante el entrenamiento. Esto no solo optimiza la memoria, sino que también introduce una ligera variación en los datos (data augmentation) para mejorar la generalización del modelo.

### Construcción del Modelo

Definimos una Red Neuronal Convolucional (CNN) simple utilizando `tf.keras` que consta de las siguientes capas:

* Capas convolucionales para extraer características.
* Capas de pooling para reducir la dimensionalidad.
* Capas de dropout para regularización.
* Capas densas para la clasificación.

### Compilación del Modelo

El modelo se compila con los siguientes parámetros:

* **Optimizador:** Adam
* **Función de Pérdida:** Categorical Crossentropy
* **Métricas:** Accuracy

### Entrenamiento del Modelo

Entrenamos el modelo utilizando los datos de entrenamiento y validación. Durante el entrenamiento, monitorizamos la precisión y la pérdida en ambos conjuntos de datos para ajustar los parámetros y prevenir el sobreajuste.

### Evaluación del Modelo y Resultados

Finalmente, como parte de esta entrega evaluamos el rendimiento del modelo en el conjunto de prueba y usamos herramientas como la matriz de confusión para visualizar el rendimiento del modelo en las diferentes categorías.

[Dataset descarga](https://drive.google.com/drive/folders/1r0ZOgscMZwfoSkpIvYU14Zf-jW9toEf1?usp=sharing)

[Estado del Arte](https://www.sciencedirect.com/science/article/pii/S1877050918309335)
