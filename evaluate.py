import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model

img_width, img_height = 224, 224
batch_size = 64

test_data_dir = '/Users/s.a.o/Documents/TEC-8/AA/M2_IA/reto/tom_and_jerry/tom_and_jerry_prep/test'

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Cargar el modelo entrenado
model = load_model("initial_model.h5")

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.n // batch_size)
print(f"Test accuracy: {test_acc}")

# Obtén las primeras 9 imágenes y etiquetas del generador de pruebas
images, labels = next(test_generator)

# Se hacen predicciones para estas imágenes
predictions = model.predict(images)

# Convierte las predicciones a clases
predicted_classes = np.argmax(predictions, axis=1)

# Convierte las etiquetas verdaderas a clases
true_classes = np.argmax(labels, axis=1)

# Imprimir el reporte de clasificación
print(classification_report(true_classes, predicted_classes))

# Crear la matriz de confusión
cm = confusion_matrix(true_classes, predicted_classes)
class_names = sorted(test_generator.class_indices.keys())

# Mostrar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Se imprimen las primeras 10 predicciones y las etiquetas verdaderas
for i in range(10):
    print(f"Image {i+1}:")
    print(f"Predicted class: {class_names[predicted_classes[i]]}")
    print(f"True class: {class_names[true_classes[i]]}")
    print()

# Predicción de una imagen nueva
image_path = '/Users/s.a.o/Documents/TEC-8/AA/M2_IA/reto/tom_and_jerry/prueba.jpeg'
image = load_img(image_path, target_size=(img_width, img_height))  # Asegúrate de que el tamaño coincida con el que usaste en el entrenamiento
image_array = img_to_array(image)  # Convierte la imagen a un array de numpy
image_array = np.expand_dims(image_array, axis=0)  # Añade una dimensión para crear un batch de tamaño 1
image_array /= 255.0  # Realiza el mismo preprocesamiento (reescalado)

# Realiza la predicción
predictions = model.predict(image_array)
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class_name = class_names[predicted_class_index]

# Mostrar la imagen y la predicción
plt.imshow(image)
plt.axis('off')
plt.title(f'Predicted class: {predicted_class_name}')
plt.show()
