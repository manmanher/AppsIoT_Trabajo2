# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("seaborn") # estilo de gráficas

# Etiquetas de las actividades

LABELS = ['primera','segunda','tercera','derecha','izquierda']

# El número de pasos dentro de un segmento de tiempo
TIME_PERIODS = 40

# Los pasos a dar de un segmento al siguiente; si este valor es igual a
# TIME_PERIODS, entonces no hay solapamiento entre los segmentos
STEP_DISTANCE = 20

# al haber solapamiento aprovechamos más los datos
#
column_names = ['timestamp',
                    'Ax',
                    'Ay',
                    'Az',
                    'Vx',
                    'Vy',
                    'Vz']
# leer csv con los datos de las actividades y añadirle el nombre de las columnas
primera=pd.read_csv("primera.csv", header=None).T
label_primera = np.empty((primera.shape[0], 1), dtype=object)
for i in range(primera.shape[0]):
    label_primera[i] = "primera"
primera.insert(0, "activity", label_primera, True) # Añadir el nombre de columnas al DataFrame
primera = primera.rename(columns={i: column_names[i] for i in range(len(column_names))})

# Separar en train y test
df_train_primera = primera.iloc[:int(len(primera)*0.7)]
df_test_primera = primera.iloc[int(len(primera)*0.7):]

segunda=pd.read_csv("segunda.csv", header=None).T
label_segunda = np.empty((segunda.shape[0], 1), dtype=object)
for i in range(segunda.shape[0]):
    label_segunda[i] = "segunda"
segunda.insert(0, "activity", label_segunda, True) # Añadir el nombre de columnas al DataFrame
segunda = segunda.rename(columns={i: column_names[i] for i in range(len(column_names))})

# Separar en train y test
df_train_segunda = segunda.iloc[:int(len(segunda)*0.7)]
df_test_segunda = segunda.iloc[int(len(segunda)*0.7):]

tercera=pd.read_csv("tercera.csv", header=None).T
label_tercera = np.empty((tercera.shape[0], 1), dtype=object)
for i in range(tercera.shape[0]):
    label_tercera[i] = "tercera"
tercera.insert(0, "activity", label_tercera, True) # Añadir el nombre de columnas al DataFrame
tercera = tercera.rename(columns={i: column_names[i] for i in range(len(column_names))})

# Separar en train y test
df_train_tercera = tercera.iloc[:int(len(tercera)*0.7)]
df_test_tercera = tercera.iloc[int(len(tercera)*0.7):]

cuarta=pd.read_csv("derecha.csv", header=None).T
label_cuarta = np.empty((cuarta.shape[0], 1), dtype=object)
for i in range(cuarta.shape[0]):
    label_cuarta[i] = "derecha"
cuarta.insert(0, "activity", label_cuarta, True) # Añadir el nombre de columnas al DataFrame
cuarta = cuarta.rename(columns={i: column_names[i] for i in range(len(column_names))})

# Separar en train y test
df_train_cuarta = cuarta.iloc[:int(len(cuarta)*0.7)]
df_test_cuarta = cuarta.iloc[int(len(cuarta)*0.7):]

quinta=pd.read_csv("izquierda.csv", header=None).T
label_quinta = np.empty((quinta.shape[0], 1), dtype=object)
for i in range(quinta.shape[0]):
    label_quinta[i] = "izquierda"
quinta.insert(0, "activity", label_quinta, True) # Añadir el nombre de columnas al DataFrame
quinta = quinta.rename(columns={i: column_names[i] for i in range(len(column_names))})

# Separar en train y test
df_train_quinta = quinta.iloc[:int(len(quinta)*0.7)]
df_test_quinta = quinta.iloc[int(len(quinta)*0.7):]

# Concatenamos los datos en un solo array con pandas para mantener el nombre de las columnas
df_train = pd.concat([df_train_primera, df_train_segunda, df_train_tercera, df_train_cuarta, df_train_quinta])
df_test = pd.concat([df_test_primera, df_test_segunda, df_test_tercera, df_test_cuarta, df_test_quinta])

#%%

# Datos que tenemos
print(df_train.shape)
print(df_test.shape)
print(df_train.info())
print(df_test.info())
# Mostramos los primeros datos
print(df_train.head())
print(df_test.head())
# Mostramos los últimos
print(df_train.tail())
print(df_test.tail())

#%% Visualizamos la cantidad de datos que tenemos
# de cada actividad 

#actividades = data['activity'].value_counts()
#plt.bar(range(len(actividades)), actividades.values)
#plt.xticks(range(len(actividades)), actividades.index)

#%% Codificamos la actividad de manera numérica

from sklearn import preprocessing

LABEL = 'ActivityEncoded'
# Transformar las etiquetas de String a Integer mediante LabelEncoder
le = preprocessing.LabelEncoder()

# Añadir una nueva columna al DataFrame existente con los valores codificados
df_train[LABEL] = le.fit_transform(df_train['activity'].values.ravel())
df_test[LABEL] = le.fit_transform(df_test['activity'].values.ravel())

print(df_train.head())
print(df_test.head())

#%% Normalizamos los datos

df_train["Ax"] = (df_train["Ax"] - min(df_train["Ax"].values)) / (max(df_train["Ax"].values) - min(df_train["Ax"].values))
df_train["Ay"] = (df_train["Ay"] - min(df_train["Ay"].values)) / (max(df_train["Ay"].values) - min(df_train["Ay"].values))
df_train["Az"] = (df_train["Az"] - min(df_train["Az"].values)) / (max(df_train["Az"].values) - min(df_train["Az"].values))
df_train["Vx"] = (df_train["Vx"] - min(df_train["Vx"].values)) / (max(df_train["Vx"].values) - min(df_train["Vx"].values))
df_train["Vy"] = (df_train["Vy"] - min(df_train["Vy"].values)) / (max(df_train["Vy"].values) - min(df_train["Vy"].values))
df_train["Vz"] = (df_train["Vz"] - min(df_train["Vz"].values)) / (max(df_train["Vz"].values) - min(df_train["Vz"].values))

df_test["Ax"] = (df_test["Ax"] - min(df_test["Ax"].values)) / (max(df_test["Ax"].values) - min(df_test["Ax"].values))
df_test["Ay"] = (df_test["Ay"] - min(df_test["Ay"].values)) / (max(df_test["Ay"].values) - min(df_test["Ay"].values))
df_test["Az"] = (df_test["Az"] - min(df_test["Az"].values)) / (max(df_test["Az"].values) - min(df_test["Az"].values))
df_test["Vx"] = (df_test["Vx"] - min(df_test["Vx"].values)) / (max(df_test["Vx"].values) - min(df_test["Vx"].values))
df_test["Vy"] = (df_test["Vy"] - min(df_test["Vy"].values)) / (max(df_test["Vy"].values) - min(df_test["Vy"].values))
df_test["Vz"] = (df_test["Vz"] - min(df_test["Vz"].values)) / (max(df_test["Vz"].values) - min(df_test["Vz"].values))


#%% Representamos para ver que se ha hecho bien

plt.figure(figsize=(5,5))
plt.plot(df_train["Ax"].values[:80])
plt.xlabel("Tiempo")
plt.ylabel("Acel X")


#%% Creamos las secuencias

from scipy import stats

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html

def create_segments_and_labels(df, time_steps, step, label_name):

    # x, y, z acceleraciones
    N_FEATURES = 6
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['Ax'].values[i: i + time_steps]
        ys = df['Ay'].values[i: i + time_steps]
        zs = df['Az'].values[i: i + time_steps]
        vxs = df['Vx'].values[i: i + time_steps]
        vys = df['Vy'].values[i: i + time_steps]
        vzs = df['Vz'].values[i: i + time_steps]
        # Lo etiquetamos como la actividad más frecuente 
        label = stats.mode(df[label_name][i: i + time_steps])[0] #Se hace la moda porque puede haber un sector que haya algunas etiquetas que estén mezcladas.
        segments.append([xs, ys, zs, vxs, vys, vzs])
        labels.append(label)

    # Los pasamos a vector
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES) # (tantas filas como hagan falta,80,3)
    labels = np.asarray(labels)

    return reshaped_segments, labels

x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

x_test, y_test = create_segments_and_labels(df_test,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

#%% observamos la nueva forma de los datos (80, 3)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)

#%% datos de entrada de la red neuronal

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

#%% transformamos los datos a flotantes

x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

x_test = x_test.astype('float32')
#y_test = y_test.astype('float32')

#%% Realizamos el one-hote econding para los datos de salida

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
y_train_hot = cat_encoder.fit_transform(y_train.reshape(len(y_train),1))
y_train = y_train_hot.toarray()

#%% RED NEURONAL

# cnn model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.utils import to_categorical

filters = 128
'''
Con esta tenemos un 96 de entrenamiento y un 40 de test

model_m = Sequential()
model_m.add(Conv1D(filters=filters, kernel_size=5, activation='relu', input_shape=(TIME_PERIODS,num_sensors)))
model_m.add(Conv1D(filters=filters/2, kernel_size=5, activation='relu'))
model_m.add(Dropout(0.5))
model_m.add(MaxPool1D(pool_size=2))
model_m.add(Flatten())
model_m.add(Dense(1000, activation='relu'))
model_m.add(Dense(num_classes, activation='softmax'))
model_m.summary()
'''


model_m = Sequential()
model_m.add(Conv1D(filters=filters, kernel_size=5, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(filters=filters, kernel_size=5, activation='relu'))
model_m.add(MaxPooling1D(pool_size=2))
model_m.add(Conv1D(filters=filters//2, kernel_size=3, activation='relu'))
model_m.add(Conv1D(filters=filters//2, kernel_size=3, activation='relu'))
model_m.add(MaxPooling1D(pool_size=2))
model_m.add(Dropout(0.5))
model_m.add(Flatten())
model_m.add(Dense(1000, activation='relu'))
model_m.add(Dense(500, activation='relu'))  # Additional dense layer
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))

model_m.summary()



#%% Guardamos el mejor modelo y utilizamos early stopping

callbacks_list = [ # ESta funcion guarda el modelo con los parámetros que se hayan entrenado hasta un numero x de epochs
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
]

#%% determinamos la función de pérdida, optimizador y métrica de funcionamiento 

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])


#%% Entrenamiento

# Batch size de la mitad de los datos de entrenamiento
BATCH_SIZE = x_train.shape[0] // 4
EPOCHS = 50

history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.1,
                      verbose=1)

#%% Visualización entrenamiento

from sklearn.metrics import classification_report

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

#%% Evaluamos el modelo en los datos de test

# actualizar dependiendo del nombre del modelo guardado
#model = keras.models.load_model("best_model.09-0.51.h5")

y_test_hot = cat_encoder.fit_transform(y_test.reshape(len(y_test),1))
y_test = y_test_hot.toarray()

test_loss, test_acc = model_m.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

#%%
# Print confusion matrix for training data
y_pred_train = model_m.predict(x_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
max_y_train = np.argmax(y_train, axis=1)
print(classification_report(max_y_train, max_y_pred_train))

 #%%
import seaborn as sns
from sklearn import metrics

def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

y_pred_test = model_m.predict(x_test)
# Toma la clase con la mayor probabilidad a partir de las predicciones de la prueba
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))