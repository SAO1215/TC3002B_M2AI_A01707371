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

El dataset fue creado por BALA BASKAR y obtenido de Kaggle, [Tom and Jerry Image classification](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification).

### Estructura y Re-diseño del dataset

Se rediseñó la estructura del dataset con el proposito de obtener una distribución equitativa en clases y evitar desbalances que podrían llevar a un rendimiento preferencial para clases específicas. Ahora, el conjunto de datos utilizado para entrenar, validar y probar el modelo está organizado en tres subconjuntos principales y se divide en cuatro categorías de clases.

Los detalles de cada subconjunto y la distribución de clases son los siguientes:

* Train (Entrenamiento): Utilizado para entrenar el modelo, contiene un total de 3160 imágenes repartidas equitativamente en las cuatro clases. Este conjunto es esencial para que el modelo aprenda las características distintivas de cada clase.
* Validation (Validación): Compuesto por 903 imágenes, este conjunto se usa para validar el rendimiento del modelo durante el entrenamiento. Permite ajustar los parámetros del modelo sin utilizar el conjunto de prueba, lo que ayuda a evitar el sobreajuste.
* Test (Prueba): Contiene 452 imágenes y se utiliza para evaluar la capacidad del modelo de generalizar a nuevos datos, es decir, datos que no se han visto durante la fase de entrenamiento.

[Enlace al Dataset usado en este proyecto](https://drive.google.com/drive/folders/1LnohhlfKUx_iZhzmpwWX5ktAFWtwON_C?usp=drive_link)

### Preprocesamiento

El preprocesamiento de las imágenes es un paso crucial para preparar nuestros datos para el entrenamiento del modelo. Este proceso incluye:

* **Redimensionamiento de Imágenes:** Todas las imágenes se redimensionan a 224x224 píxeles.
* **Normalización:** Los valores de píxeles se escalan para estar en el rango [0, 1] para mejorar la convergencia durante el entrenamiento.
* **División de Datos:** El conjunto de datos se divide en 70% para entrenamiento, 20% para validación y 10% para prueba utilizando la biblioteca `splitfolders`.

### Carga de Datos con TensorFlow y Keras

Se usa `tf.keras.preprocessing.image.ImageDataGenerator` para cargar y transformar las imágenes en lotes, facilitando el manejo durante el entrenamiento. Esto no solo optimiza la memoria, sino que también introduce una ligera variación en los datos (data augmentation) para mejorar la generalización del modelo.

### Construcción del Modelo

En este proyecto, hemos construido un modelo de Red Neuronal Convolucional (CNN) utilizando la biblioteca `tf.keras`. El modelo consta de las siguientes capas:

* **Carga del Modelo Base:** Utilizamos la arquitectura MobileNetV2 como nuestro modelo base, preentrenado en el conjunto de datos _ImageNet_. Esto nos proporciona una sólida base de características visuales sin necesidad de entrenamiento adicional en nuestros datos específicos. La capa `include_top=False` significa que excluimos las capas densas finales del modelo base, lo que nos permite añadir nuestras propias capas personalizadas para adaptar el modelo a nuestro problema específico. Además, definimos el tensor de entrada utilizando `input_tensor` y congelamos los pesos del modelo base para evitar que se actualicen durante el entrenamiento estableciendo `base_model.trainable = False`.
* **Capa de Global Average Pooling:** Después de la salida del modelo base, agregamos una capa de _Global Average Pooling_ para reducir la dimensionalidad de las características extraídas y generar una representación compacta de la información visual.
* **Capas Densas Adicionales:** A continuación, añadimos capas densas adicionales para la clasificación. Comenzamos con una capa densa con 512 unidades y función de activación _ReLU_, que ayuda a aprender representaciones no lineales en los datos. Finalmente, agregamos una capa densa de salida con 4 unidades y función de activación _softmax_, que produce una distribución de probabilidad sobre las clases objetivo, permitiendo la clasificación de las entradas en una de las 4 categorías posibles.

#### Compilación del Modelo

El modelo se compila con los siguientes parámetros:

* **Optimizador:** Adam
* **Función de Pérdida:** Categorical Crossentropy
* **Métricas:** Accuracy

### Reafinamiento del Modelo

Se usa la misma base de la construccion del model, pero se incorporan:

* **Dropout:** Se ha agregado una capa de Dropout con una tasa del 30%. El Dropout es una técnica de regularización que ayuda a prevenir el sobreajuste al apagar aleatoriamente un cierto porcentaje de unidades durante el entrenamiento, lo que obliga al modelo a aprender representaciones más robustas y generalizables.
* **Programación del Ritmo de Aprendizaje:** Se utiliza una programación del ritmo de aprendizaje exponencial para ajustar dinámicamente la tasa de aprendizaje durante el entrenamiento.
* **Ajuste Fino (Fine-tuning):** Se aplica ajuste fino al modelo, descongelando algunas capas del modelo base para permitir que se adapten mejor a los datos específicos del proyecto.
* **Recompilar el Modelo:** Después de ajustar las capas del modelo base, se vuelve a compilar el modelo con un nuevo optimizador y una tasa de aprendizaje específica.

#### Parámetros Utilizados en el Reafinamiento

* initial_learning_rate: La tasa de aprendizaje inicial utilizada para entrenar el modelo. En este ejemplo, se ha establecido en 1e-4.
* lr_schedule: Un objeto ExponentialDecay que ajusta la tasa de aprendizaje durante el entrenamiento. La tasa de aprendizaje se reduce exponencialmente a medida que avanza el entrenamiento para mejorar la convergencia del modelo.
* fine_tune_at: El número de capas en el modelo base que se permiten entrenar durante el reafinamiento. En este ejemplo, se han congelado las primeras 50 capas del modelo base, y solo las capas posteriores se ajustan durante el reafinamiento.

### Entrenamiento del Modelo

Entrenamos el modelo, en 20 epocas, utilizando los datos de entrenamiento y validación. Durante el entrenamiento, monitorizamos la precisión y la pérdida en ambos conjuntos de datos para ajustar los parámetros y prevenir el sobreajuste.

### Evaluación del Modelo y Resultados

Después del Reafinamiento, el modelo alcanzó una precisión de prueba del 89%.

#### Anexos

[Articulo 1](https://www.sciencedirect.com/science/article/pii/S1877050918309335)

[Articulo 2](https://ieeexplore.ieee.org/document/9422058)

[Articulo 3](https://www.sciencedirect.com/science/article/pii/S157495412300482X)

[Articulo 4](https://www.sciencedirect.com/science/article/pii/S1877050921023565)

[Articulo 5](https://pdf.sciencedirectassets.com/776627/1-s2.0-S2666285X21X00039/1-s2.0-S2666285X21000960/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEE8aCXVzLWVhc3QtMSJHMEUCIDKiMH6gCKQW%2BQOPApbU%2BmitC%2FdunX8%2BV7Da3ids%2BLsqAiEAxtsB9CTg8%2B%2FVkiFLzbMTGOWyMMzUDHbdSGtTCBxx3M8quwUIl%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDMJyc42jQJ5eHJTKiSqPBUOXirtzeFNYmVbMr9i8XyS%2FwuqsZ3aC75naAhEFLKNR927oXT9J2d55puGvts%2Bc%2BoJGULloHVyFWQ3QjHNwSErzqF%2BTAh5iWFPlgSZzxfL9c9Zmd0Q3JukzD9fOzlAuLdI3XheX09tSxzJ%2B1z4TLXdgfVa%2BM%2BXeqeqTNpbO3MAmTpSSXVvGFJvW%2Faw%2BfuAFJMzI%2BV4uGP21q7aiyuxK6PLRSmfeG0yxozqTP3wQCtpBQ9Yf1khzg7vCL%2F4mPEoA%2FYY8ODBLuf3du5pr5y9FlmIETOzYxscSBFLNcjxH7EI%2FbcD4q0XQidavtcDXLeejUtv9B27QS5jIIheADKs8lRF2Gv8Di%2FPDRoRFV%2BAVEONxWliyb8hxzRlGbBBZXLLw0XH3ZvpGXfZulIeJT1U642B5KJW2sO%2BUhCGqZwsEUqEG33UGqQW147VAS9R7uMisl8mw0e9EWwJN2xVZHvtLXmeiHZ%2FrFKnfDo70pDZ3ZwuDQIxZgcPHW7XZUQo0jjF03NCNFgvhju6JoLcXjQ8BtZlWpv5%2Bid%2FsY41MZMafsX1M%2Fdk8%2BUx%2B%2BlxRt2ncdyw6G8pgEMcPS6JFZC2OV6RzIXIcDXbkMjmFUCc%2FoGeQFRvZZMwFGzFxXegRBoKLPWa0MFldHrvYPuNxXE4CIhPM6hszX2XMSR9Ro6NjfQxgfrJhIxYAICMwerylnPBdmHlURjHOuY0VJcSk%2Bphjot99fEj1JFu4gZGNSDkilfXqpWdtegxulxKnpinFcQz2eSs0MBAaiDH6YncsETj2g9ReeB07I2%2F2c%2Fep%2BQgU1qpujeyzDGgmDbBA6%2BzoPdnhNXkbgji%2BVYTKGAfUVvOlWTnBwYYWOewy7RsS2Qrf0uiiJHcwnoemsQY6sQF39CczUgfPNjQPFpNtxHoHHXiElLvtbmFPDy05NYpAa2kLGOGR4vhtWDNT1MEDTtD%2FFHQnqIpDNzQXv7KxN8ego6NjrM%2BN5i4LMXQnkT5VHak%2Fv5StI%2BH8D6e4lSUiLlFrkOJgX1CsnlcNkTYEa6Rptlwer2cEjTTH9I43KdX%2F3yrxqtCWSRZhLsG2qid1C%2FDeh0JiDWOGSKnQ4a7l%2BkFaCuevBy%2FdUz2068CsOBGAqiM%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240424T225602Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRAJPI23H%2F20240424%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ddfcdcd90c939495dce6893841a8019c2b13275dcb3e788732112cf7325757d0&hash=ffd772c756a07f5e9e9b3d2ea08fe26c17ee18a6262ea83ad36108bb0f52fffb&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2666285X21000960&tid=spdf-127f923c-5f0e-4344-ace3-3b732f21ffdd&sid=86bb57805f8b8743914af9797d691c61c505gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0b1c57585e5f0f095105&rr=8799af2d88702e5d&cc=mx)

[Articulo 6](https://www.ijisae.org/index.php/IJISAE/article/view/2594/1176)
