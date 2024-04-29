# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Ruta a los directorios que contienen los conjuntos de datos de entrenamiento, validación y prueba
train_data_dir = '/Users/s.a.o/Documents/TEC-8/AA/M2_IA/reto/tom_and_jerry/tom_and_jerry_prep/train'
valid_data_dir = '/Users/s.a.o/Documents/TEC-8/AA/M2_IA/reto/tom_and_jerry/tom_and_jerry_prep/val'
test_data_dir = '/Users/s.a.o/Documents/TEC-8/AA/M2_IA/reto/tom_and_jerry/tom_and_jerry_prep/test'

img_width, img_height = 224, 224
batch_size = 64

# Definición de los generadores de datos por cada conjunto
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
valid_test_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Creación de los generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

valid_generator = valid_test_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Definir la entrada
input_tensor = layers.Input(shape=(img_width, img_height, 3))

# Cargar el modelo base
base_model = MobileNetV2(include_top=False, input_tensor=input_tensor, weights='imagenet')
base_model.trainable = False

# Agregar la capa de Global Average Pooling
x = layers.GlobalAveragePooling2D()(base_model.output)
# Agregar capas densas adicionales
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Dropout(0.3)(x)
output_tensor = layers.Dense(4, activation='softmax')(x)

# Construir el modelo
model = Model(inputs=input_tensor, outputs=output_tensor)

# Ajustar la taza de aprendizaje
initial_learning_rate = 1e-4

lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9
)

# Compila el modelo con el optimizador especificando
optimizer = optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Ajuste Fino (Fine-tuning)
base_model.trainable = True
fine_tune_at = 50

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compilar el modelo (necesario para que los cambios tengan efecto)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Visualizar la estructura del modelo
model.summary()

# Entrena el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
)

# Se evalúa el modelo en el conjunto de prueba y se imprime la precisión obtenida.
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.n // batch_size)
print(f"Test accuracy: {test_acc}")

# Se recuperan las métricas de entrenamiento y validación a lo largo de las épocas para su posterior visualización.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# Visualizar el rendimiento del modelo durante el entrenamiento
plt.plot(epochs, acc, color = "#222831", label='Training accuracy')
plt.plot(epochs, val_acc, color = "#36D1DC", label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, color = "#222831", label='Training loss')
plt.plot(epochs, val_loss, color = "#36D1DC", label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Se realizan predicciones en el conjunto de prueba
predictions = model.predict(test_generator, steps=(test_generator.n // test_generator.batch_size) + 1)
predicted_classes = np.argmax(predictions, axis=-1)

true_classes = test_generator.classes

if len(predicted_classes) > len(true_classes):
    predicted_classes = predicted_classes[:len(true_classes)]

# Calcula la matriz de confusión y se imprime un informe de clasificación para evaluar el rendimiento del modelo
cm = confusion_matrix(true_classes, predicted_classes)
print(classification_report(true_classes, predicted_classes))

# Visualizar la matriz de confusión 
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=list(test_generator.class_indices.keys()), yticklabels=list(test_generator.class_indices.keys()))
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# El modelo se guarda en un archivo HDF5 para su uso futuro.
model.save("final_model.h5")