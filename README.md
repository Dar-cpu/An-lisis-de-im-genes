# Análisis de Imágenes
**Este proyecto pretende analizar imágenes desde el dominio de la frecuencia y preparar conjuntos de datos para el entrenamiento de modelos de super-resolución de imágenes (primera versión).**
## 1.- Estructura del Proyecto 
- a) El sistema se compone de tres archivos principales: Empezando por el "Dataseet.py" que en teoría es un Generador de Dataset para Super-Resolución que permite generar imágenes de baja resolución a partir de imágenes de alta resolución, útiles para entrenar modelos de super-resolución (aún por solucionar algunos errores).
- b) El análisis Espectral de Imágenes "Espectros.py" que permite visualizar la Transformada de Fourier 2D de una imagen, mostrando el espectro de magnitud, espectro de fase, parte real e imaginaria, reconstrucción del espectro y un gráfico de frecuencia en una línea central.
- c)  El punto de Entrada del Sistema "Main.py" es la interfaz gráfica principal que permite al usuario cargar imágenes y visualizar su análisis usando distintos modos.

## 2.- Requsitos:
- Python 3.8+
#### Librerías:
- opencv-python
- numpy
- matplotlib
- tkinter
- tensorflow

## Notas 
*El proyecto aún contiene errores leves y moderados que pueden ser solucionados acorde a la aplicación*
- Si tienes alguna duda usa issues 

##### Instalación rápida:
<pre> ´´´pip install opencv-python numpy matplotlib tensorflow´´´  <pre>

  
