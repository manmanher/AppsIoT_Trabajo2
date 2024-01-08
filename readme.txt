Autores: Manuel Manzano (@manmanher) y Fernando Garrucho (fergarfer1).
Este repositorio contiene los códigos y dataset recopilado para el 2º Trabajo de la asignatura Aplicaciones IoT, del Máster Universitario de Ingeniería de Telecomunicación de la Universidad de Sevilla.
- En la carpeta Dataset se encuentran las medidas realizadas con la Raspberry Pi y el sensor SenseHat de las 5 clases.
- En la carpeta Code el código python, diferenciandose los siguientes dos archivos.
	- primera.py: El código ejecutado en la Rpi para recopilar los datos, fue ejecutados 5 veces, una por clase, realizando el movimiento correspondiente de forma repetida en cada ejecución.
	- CNN1D.py:   El código de la red neuronal donde se procesa el dataset generado, se entrena la red neuronal y se hacen los tests correspondientes para calcular su Accuracy.