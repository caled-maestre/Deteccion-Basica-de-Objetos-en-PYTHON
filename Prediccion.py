#---- Importar Librerias ----
import tensorflow as tf
import cv2
import numpy as np
from keras_preprocessing.image import (img_to_array)

#---- Direcciones de los modelos ----
ModeloDenso = r'C:\Users\CALED\PycharmProjects\Deteccion de objetos\.venv\ClasificadorDenso.h5'
ModeloCNN = r'C:\Users\CALED\PycharmProjects\Deteccion de objetos\.venv\ClasificadorCNN.h5'
ModeloCNN2 = r'C:\Users\CALED\PycharmProjects\Deteccion de objetos\.venv\ClasificadorCNN2.h5'

#---- Leer las redes neuronales ----
#Denso
Denso = tf.keras.models.load_model(ModeloDenso)
pesosDenso = Denso.get_weights()
Denso.set_weights(pesosDenso)
#CNN
CNN = tf.keras.models.load_model(ModeloCNN)
pesosCNN = CNN.get_weights()
CNN.set_weights(pesosCNN)
#CNN2
CNN2 = tf.keras.models.load_model(ModeloCNN2)
pesosCNN2 = CNN2.get_weights()
CNN2.set_weights(pesosCNN2)

#---- Realizar la VideoCaptura ----
cap = cv2.VideoCapture(0)

#Empieza el While True
while True:
    #Lectura de Videocaptura
    ret, frame = cap.read()

    #pasar a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Redimensionar las imagenes
    gray = cv2.resize(gray, (200,200), interpolation = cv2.INTER_CUBIC)

    #Normalizamos la imagen
    gray = np.array(gray).astype(float) / 255

    #Convertir la imagen en matriz
    img = img_to_array(gray)
    img = np.expand_dims(img, axis=0)

    #Realizar la prediccion
    prediccion = CNN.predict(img)
    prediccion = prediccion[0]
    prediccion = prediccion[0]
    print(prediccion)

    #Realizar la clasificacion
    if prediccion <= 0.5:
        cv2.putText(frame, "Gato", (200, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Perro", (200, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    #Mostrar los fotogramas
    cv2.imshow("CNN", frame)

    t = cv2.waitKey(1)
    if t == 27:
        break
cv2.destroyAllWindows()
cap.release()