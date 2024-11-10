#---- Importar Librerias ----
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#---- Almacenar la direccion de las imagenes ----
entrenamiento = r'C:\Users\CALED\PycharmProjects\Deteccion de objetos\.venv\Datasets\Entrenamiento'
validacion = r'C:\Users\CALED\PycharmProjects\Deteccion de objetos\.venv\Datasets\Validacion'

listaTrain = os.listdir(entrenamiento)
listaTest = os.listdir(validacion)

 # ---- Parametros de las imagenes ----
ancho, alto = 200,200
#Listas Entrenamiento
etiquetas = []
fotos = []
datos_train = []
con = 0
#Listas Validacion
etiquetas2 = []
fotos2 = []
datos_vali = []
con2 = 0

 # ---- Extraer en una lista las fotos y entra las etiquetas ----
#Entrenamiento
for nameDir in listaTrain:
    nombre = entrenamiento + '/' + nameDir #Leemos las fotos

    for fileName in os.listdir(nombre): #Asignar las etiquetas a cada foto
	    etiquetas.append(con) #Valor de la etiqueta( asignamos 0 a la primera etiqueta y 1 a la segunda
	    img = cv2.imread(nombre + '/' + fileName,0) #Leemos la imagen
	    img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC) #Redimensionar las imagenes
	    img = img.reshape(ancho,alto,1) # Dejar 1 solo canal
	    datos_train.append([img,con])
	    fotos.append(img) #Añadir las imagenes de EDG

    con = con + 1

#Validacion
for nameDir2 in listaTest:
    nombre2 = validacion + '/' + nameDir2 #Leemos las fotos

    for fileName2 in os.listdir(nombre2): #Asignar las etiquetas a cada foto
	    etiquetas2.append(con2) #Valor de la etiqueta( asignamos 0 a la primera etiqueta y 1 a la segunda
	    img2 = cv2.imread(nombre2 + '/' + fileName2,0) #Leemos la imagen
	    img2 = cv2.resize(img2, (ancho,alto), interpolation=cv2.INTER_CUBIC) #Redimensionar las imagenes
	    img2 = img2.reshape(ancho,alto,1) # Dejar 1 solo canal
	    datos_vali.append([img2,con2])
	    fotos2.append(img2) #Añadir las imagenes de EDG

    con2 = con2 + 1

#---- Normalizar Imagenes ----
fotos = np.array(fotos).astype(float)/255
print(fotos.shape)
fotos2 = np.array(fotos2).astype(float)/255
print(fotos2.shape)
#Pasar las listas a arrays
etiquetas = np.array(etiquetas)
etiquetas2 = np.array(etiquetas2)

imgTrainGen = ImageDataGenerator(
    rotation_range = 50, # Rotacion aleatoria de las imagenes
    width_shift_range = 0.3, # Mover la imagen a los lados
    height_shift_range = 0.3, # Mover la imagen arriba y abajo
    shear_range = 15, # Inclinar la imagen
    zoom_range = [0.5, 1.5], # Hacer zoom a la imagen
    vertical_flip = True, # Flip verticales aleatorios
    horizontal_flip = True #Flip horizontales aleatorios
)

imgTrainGen.fit(fotos)
plt.figure(figsize=(20,8))
for imagen, etiqueta in imgTrainGen.flow(fotos, etiquetas, batch_size=10, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i], cmap='gray')
    plt.show()
    break

imgTrain = imgTrainGen.flow(fotos, etiquetas, batch_size=32)

#---- Estructura de la red neuronal convolucional ----
#Modelo con capas Densas
ModeloDenso = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (200,200,1)),
    tf.keras.layers.Dense(150, activation = 'relu'),
    tf.keras.layers.Dense(150, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])

#Modelo con capas convolucionales
ModeloCNN = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200,200,1)), #Capa de entrada convolucional 32 Kernel
    tf.keras.layers.MaxPooling2D(2,2), #Capa de Max pooling
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'), #Capa convolucional de 64 kernel
    tf.keras.layers.MaxPooling2D(2,2), #Capa de Max pooling
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'), #Capa convolucional de 128 kernel
    tf.keras.layers.MaxPooling2D(2,2), #Capa de Max pooling

    #Capas Densas de clasificacion
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'), # Capa densa con 256 neuronas
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#Modelo con capas convolucionales y Drop Out
ModeloCNN2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200,200,1)), #Capa de entrada convolucional 32 Kernel
    tf.keras.layers.MaxPooling2D(2,2), #Capa de Max pooling
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'), #Capa convolucional de 64 kernel
    tf.keras.layers.MaxPooling2D(2,2), #Capa de Max pooling
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'), #Capa convolucional de 128 kernel
    tf.keras.layers.MaxPooling2D(2,2), #Capa de Max pooling

    #Capas Densas de clasificacion
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'), # Capa densa con 256 neuronas
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#---- Compilar los modelos: agregar el optimizador y la funcion de perdida ----
ModeloDenso.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

ModeloCNN.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

ModeloCNN2.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

#---- Observar y Entrenar las redes ----
#para visualizar: tensorboard --logdir= r'C:\Users\CALED\PycharmProjects\Deteccion de objetos'
#Entrenar el modelo Denso:
BoardDenso = TensorBoard(log_dir = r'C:\Users\CALED\PycharmProjects\Deteccion de objetos')
ModeloDenso.fit(imgTrain, batch_size = 32, validation_data = (fotos2, etiquetas2),
                 epochs = 100, callbacks = [BoardDenso], steps_per_epoch = int(np.ceil(len(fotos) / float(32))),
                 validation_steps = int(np.ceil(len(fotos2) / float(32))))
#Guardar el modelo
ModeloDenso.save('ClasificadorDenso.h5')
ModeloDenso.save_weights('pesosDenso.h5')
print("Terminamos el modelo Denso")

#Entrenar CNN sin DO
BoardCNN = TensorBoard(log_dir = r'C:\Users\CALED\PycharmProjects\Deteccion de objetos')
ModeloCNN.fit(imgTrain, batch_size = 32, validation_data = (fotos2, etiquetas2),
                 epochs = 100, callbacks = [BoardCNN], steps_per_epoch = int(np.ceil(len(fotos) / float(32))),
                 validation_steps = int(np.ceil(len(fotos2) / float(32))))
#Guardar el modelo
ModeloCNN.save('ClasificadorCNN.h5')
ModeloCNN.save_weights('pesosCNN.h5')
print("Terminamos el modelo CNN 1")

#Entrenar CNN con DO
BoardCNN2 = TensorBoard(log_dir = r'C:\Users\CALED\PycharmProjects\Deteccion de objetos')
ModeloCNN2.fit(imgTrain, batch_size = 32, validation_data = (fotos2, etiquetas2),
                 epochs = 100, callbacks = [BoardCNN2], steps_per_epoch = int(np.ceil(len(fotos) / float(32))),
                 validation_steps = int(np.ceil(len(fotos2) / float(32))))
#Guardar el modelo
ModeloCNN2.save('ClasificadorCNN2.h5')
ModeloCNN2.save_weights('pesosCNN2.h5')
print("Terminamos el modelo CNN 2")