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

Se utiliza TensorFlow y su módulo ImageDataGenerator para cargar y preprocesar las imágenes. Las imágenes se redimensionan a 224x224 píxeles y se normalizan antes de ser alimentadas al modelo. También se realiza un aumento de datos en el conjunto de entrenamiento para mejorar la generalización del modelo.

### Visualización

Se proporciona una visualización de las primeras imágenes del conjunto de datos, junto con sus etiquetas correspondientes, para verificar la carga y el preprocesamiento correctos de los datos.

[Dataset descarga](https://drive.google.com/drive/folders/1r0ZOgscMZwfoSkpIvYU14Zf-jW9toEf1?usp=sharing)
