# Importaciones necesarias para procesamiento de imágenes y interfaz gráfica
from email.mime import image
import os
from matplotlib.pyplot import gray
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import scipy.fftpack as fftpack
from scipy import signal
from skimage.feature import graycomatrix, graycoprops
from skimage import feature
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage, QPixmap
from Dataseet import DatasetGenerator
from PyQt5.QtWidgets import (
    QFileDialog, 
    QProgressDialog, 
    QMessageBox
)
from PyQt5.QtCore import Qt

# Clase para procesamiento de transformada de Fourier
class FourierTransformProcessor:
    @staticmethod
    def low_pass_filter(image, cutoff=30):
        """
        Filtro de paso bajo utilizando Transformada de Fourier.
        Suaviza la imagen eliminando componentes de alta frecuencia.
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            cutoff (int): Radio de corte de frecuencia
        
        Returns:
            numpy.ndarray: Imagen filtrada
        """
        # Convertir a escala de grises si es imagen a color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calcular Transformada de Fourier 2D
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        # Crear máscara de filtro de paso bajo
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), cutoff, 1, -1)
        
        # Aplicar máscara al dominio de frecuencia
        fshift_filtered = fshift * mask
        
        # Transformada de Fourier Inversa
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        
        # Normalizar y convertir de vuelta al formato de imagen original
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def high_pass_filter(image, cutoff=30):
        """
        Filtro de paso alto utilizando Transformada de Fourier.
        Realza bordes y detalles eliminando componentes de baja frecuencia.
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            cutoff (int): Radio de corte de frecuencia
        
        Returns:
            numpy.ndarray: Imagen filtrada
        """
        # Convertir a escala de grises si es imagen a color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calcular Transformada de Fourier 2D
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        # Crear máscara de filtro de paso alto
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), cutoff, 0, -1)
        
        # Aplicar máscara al dominio de frecuencia
        fshift_filtered = fshift * mask
        
        # Transformada de Fourier Inversa
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        
        # Normalizar y convertir de vuelta al formato de imagen original
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def directional_filter(image, angle=45, width=10):
        """
        Filtro direccional utilizando Transformada de Fourier.
        Resalta características en una orientación específica.
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            angle (float): Ángulo de filtrado direccional (en grados)
            width (int): Ancho de la banda direccional
        
        Returns:
            numpy.ndarray: Imagen filtrada
        """
        # Convertir a escala de grises si es imagen a color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calcular Transformada de Fourier 2D
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        # Crear máscara de filtro direccional
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Crear máscara que selecciona frecuencias a lo largo de un ángulo específico
        mask = np.zeros((rows, cols), np.uint8)
        
        # Convertir ángulo a radianes
        theta = np.deg2rad(angle)
        
        # Crear máscara rectangular rotada
        rr, cc = np.indices((rows, cols))
        rotated_mask = np.abs(np.cos(theta) * (cc - ccol) + np.sin(theta) * (rr - crow)) <= width
        
        mask[rotated_mask] = 1
        
        # Aplicar máscara al dominio de frecuencia
        fshift_filtered = fshift * mask
        
        # Transformada de Fourier Inversa
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        
        # Normalizar y convertir de vuelta al formato de imagen original
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def laplacian_enhancement(image):
        """
        Realce de imagen utilizando el operador Laplaciano.
        Resalta cambios rápidos en la intensidad de los píxeles.
        
        Args:
            image (numpy.ndarray): Imagen de entrada
        
        Returns:
            numpy.ndarray: Imagen realzada
        """
        # Convertir a escala de grises si es imagen a color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Aplicar filtro Laplaciano
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Normalizar y convertir a uint8
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Realzar imagen original con Laplaciano
        enhanced = cv2.addWeighted(gray, 1.5, laplacian, -0.5, 0)
        
        # Convertir de vuelta a 3 canales si la original era a color
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
# Clase de Super Resolución 
class SuperResolucion:
    def __init__(self):
        # Configuración sin uso de GPU específico
        tf.config.set_visible_devices([], 'GPU')
        
        # Usar entrenamiento de precisión mixta (opcional, pero funciona en CPU)
        tf.keras.mixed_precision.set_global_policy('float32')
        
        # Ruta para guardar el modelo
        self.model_path = "modelo_super_resolucion.h5"
        self.dataset_generator = DatasetGenerator()
        
        # Construir modelo de red neuronal
        self.model = self._construir_modelo_sr()

    def _construir_modelo_sr(self):
        """
        Arquitectura mejorada de Modelo de Super Resolución
        Usa más capas y conexiones residuales para mejor rendimiento
        """
        # Definir forma de entrada flexible
        input_shape = (None, None, 3)
        inputs = Input(shape=input_shape)

        # Modelo mejorado con aprendizaje residual
        x = Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
    
        # Bloque residual
        residual = x
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same')(x)
        x = tf.keras.layers.Add()([residual, x])
        x = tf.keras.layers.Activation('relu')(x)
    
        # Capas de escalado
        x = Conv2DTranspose(128, 3, strides=2, padding='same')(x)
        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = Conv2D(3, 3, padding='same')(x)

        # Crear modelo
        model = Model(inputs, x)
    
        # Optimizador adaptativo con tasa de aprendizaje
        optimizador = tf.keras.optimizers.Adam(
            learning_rate=1e-4, 
            clipnorm=1.0  # Recorte de gradiente para prevenir explosión
        )
    
        # Usar pérdida perceptual para mejor calidad visual
        model.compile(
            optimizer=optimizador, 
            loss='mse',  # Pérdida de error cuadrático medio
            metrics=['mae']  # Métrica de error absoluto medio
        )
    
        return model  
    def entrenar_desde_directorio(self, directorio: str, epocas: int = 10):
        """
        Método unificado para generar dataset y entrenar desde un directorio
        
        Args:
            directorio (str): Ruta al directorio con imágenes de entrenamiento
            epocas (int): Número de épocas para entrenar
        """
        try:
            # Limpiar y regenerar dataset
            self.dataset_generator = DatasetGenerator()
            
            # Generar imágenes para el dataset
            imagenes_procesadas = self.dataset_generator.generar_imagenes_desde_directorio(
                directorio, 
                scale_factors=[0.5, 0.75]  # Puedes ajustar estos factores
            )
            
            if imagenes_procesadas == 0:
                print("No se procesaron imágenes. Verifica el directorio y los archivos.")
                return
            
            # Obtener datos de entrenamiento
            try:
                X_train, Y_train = self.dataset_generator.obtener_datos_entrenamiento()
                
                # Verificación adicional de datos
                if X_train.size == 0 or Y_train.size == 0:
                    print("No se encontraron imágenes válidas para entrenamiento.")
                    return
                
                print(f"Preparando entrenamiento con {len(X_train)} imágenes")
                
                # Configurar callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', 
                        patience=5, 
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', 
                        factor=0.5, 
                        patience=3, 
                        min_lr=1e-6
                    )
                ]
                
                # Entrenar modelo
                historia = self.model.fit(
                    X_train, Y_train, 
                    epochs=epocas, 
                    batch_size=32, 
                    validation_split=0.2,
                    callbacks=callbacks
                )
                
                # Guardar modelo
                self.model.save(self.model_path)
                print(f"Modelo guardado en {self.model_path}")
                
                # Graficar historia de entrenamiento (opcional)
                self._graficar_historia(historia)
                
            except Exception as e:
                print(f"Error al obtener datos de entrenamiento: {e}")
                import traceback
                traceback.print_exc()
        
        except Exception as e:
            print(f"Error general durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
    
    def _graficar_historia(self, historia):
        """
        Graficar la historia de entrenamiento
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12,4))
        
        # Gráfico de pérdida
        plt.subplot(1,2,1)
        plt.plot(historia.history['loss'], label='Pérdida de Entrenamiento')
        plt.plot(historia.history['val_loss'], label='Pérdida de Validación')
        plt.title('Modelo - Pérdida')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        # Gráfico de precisión
        plt.subplot(1,2,2)
        plt.plot(historia.history.get('accuracy', []), label='Precisión de Entrenamiento')
        plt.plot(historia.history.get('val_accuracy', []), label='Precisión de Validación')
        plt.title('Modelo - Precisión')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('historia_entrenamiento.png')
        plt.close()
    
    def aplicar_super_resolucion(self, imagen):
        """Método de super resolución optimizado para CPU"""
        # Convertir a float32 y normalizar
        imagen = imagen.astype(np.float32) / 255.0
        
        # Redimensionar si es demasiado grande
        dim_max = 1024
        altura, anchura = imagen.shape[:2]
        factor_escala = min(dim_max / altura, dim_max / anchura)
        
        if factor_escala < 1:
            imagen = cv2.resize(imagen, None, fx=factor_escala, fy=factor_escala)
        
        # Expandir dimensiones para predicción del modelo
        tensor_imagen = tf.convert_to_tensor(imagen[np.newaxis, ...])
        
        # Predicción con CPU
        mejorada = self.model.predict(tensor_imagen)[0]
        
        # Post-procesamiento
        mejorada = np.clip(mejorada, 0, 1)
        mejorada = (mejorada * 255).astype(np.uint8)
        
        return mejorada
        #pass

# Clase de Reducción de Ruido 
class NoiseReduction:
    @staticmethod
    def reduce_noise(image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
        # Verificar si la imagen es a color o escala de grises
        if len(image.shape) == 3:
            # Convertir a escala de grises para la transformada Laplaciana
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Reducción de ruido con Non-Local Means
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image, 
                None, 
                h, 
                hColor, 
                templateWindowSize, 
                searchWindowSize
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image, 
                None, 
                h, 
                templateWindowSize, 
                searchWindowSize
            )
        
        # Aplicar transformada Laplaciana para resaltar bordes y detalles
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Normalizar la transformada Laplaciana
        laplacian_normalized = cv2.normalize(
            laplacian, 
            None, 
            alpha=0, 
            beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        
        # Combinar la imagen denoisada con los bordes Laplacianos
        # Ajusta los pesos según necesites
        enhanced = cv2.addWeighted(
            denoised, 
            0.8,  # Peso de la imagen denoisada
            cv2.cvtColor(laplacian_normalized, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else laplacian_normalized, 
            0.2,  # Peso de los bordes Laplacianos
            0
        )
        
        return enhanced

# Clases de análisis de características 
class TextureAnalyzer:
    def __init__(self):
        self.distances = [1, 2, 3]
        self.angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    def compute_glcm_features(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        glcm = graycomatrix(image, self.distances, self.angles, 256, symmetric=True, normed=True)

        features = {
            'contrast': graycoprops(glcm, 'contrast'),
            'dissimilarity': graycoprops(glcm, 'dissimilarity'),
            'homogeneity': graycoprops(glcm, 'homogeneity'),
            'energy': graycoprops(glcm, 'energy'),
            'correlation': graycoprops(glcm, 'correlation')
        }

        return features

    def compute_lbp(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = normalize(hist.reshape(1, -1))[0]

        return hist

# Clase de análisis geométrico 
class GeometricAnalyzer:
    def extract_geometric_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            moments = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(moments)

            features.append({
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'hu_moments': hu_moments.flatten()
            })

        return features, contours

# Clase de mejora de imagen 
class ImageEnhancer:
    def correct_illumination(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced

    def adaptive_threshold(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    
class ImageCompression:
    def __init__(self):
        self.original_image = None
        self.compressed_image = None

    def compress_image_dct(self, image, compression_ratio=0.1):
        """
        Comprimir imagen usando DCT
        
        Parámetros:
        - image: imagen de entrada en formato numpy array
        - compression_ratio: ratio de compresión (0.1 = 90% de compresión)
        
        Retorna imagen comprimida
        """
        # Convertir a float32 y a YCrCb
        img_float = image.astype(np.float32)
        img_ycrcb = cv2.cvtColor(img_float, cv2.COLOR_BGR2YCrCb)
        
        # Separar canales
        y_channel, cr_channel, cb_channel = cv2.split(img_ycrcb)
        
        # Función para aplicar DCT a un canal
        def compress_channel(channel):
            # Dividir la imagen en bloques de 8x8
            h, w = channel.shape
            h_blocks = h // 8
            w_blocks = w // 8
            
            # Crear un nuevo canal para el resultado
            compressed_channel = np.zeros_like(channel)
            
            for y in range(h_blocks):
                for x in range(w_blocks):
                    # Extraer bloque 8x8
                    block = channel[y*8:(y+1)*8, x*8:(x+1)*8]
                    
                    # Aplicar DCT
                    dct_block = cv2.dct(block)
                    
                    # Calcular número de coeficientes a mantener
                    num_keep = int(64 * (1 - compression_ratio))
                    
                    # Obtener indices de los coeficientes más significativos
                    flat_dct = dct_block.flatten()
                    indices = np.abs(flat_dct).argsort()[-num_keep:]
                    
                    # Crear máscara de compresión
                    mask = np.zeros_like(flat_dct)
                    mask[indices] = 1
                    mask = mask.reshape(dct_block.shape)
                    
                    # Aplicar máscara
                    compressed_dct = dct_block * mask
                    
                    # Aplicar DCT inversa
                    compressed_block = cv2.idct(compressed_dct)
                    
                    # Guardar bloque comprimido
                    compressed_channel[y*8:(y+1)*8, x*8:(x+1)*8] = compressed_block
            
            return compressed_channel
        
        # Comprimir cada canal
        y_compressed = compress_channel(y_channel)
        cr_compressed = compress_channel(cr_channel)
        cb_compressed = compress_channel(cb_channel)
        
        # Recombinar canales
        compressed_ycrcb = cv2.merge([y_compressed, cr_compressed, cb_compressed])
        
        # Convertir de vuelta a BGR
        compressed_bgr = cv2.cvtColor(compressed_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # Normalizar y convertir a uint8
        compressed_bgr = np.clip(compressed_bgr, 0, 255).astype(np.uint8)
        
        return compressed_bgr

    def compress_and_save_image(self, compression_ratio=0.1):
        """Comprimir imagen y guardar"""
        if self.original_image is None:
            QMessageBox.warning(self, "Advertencia", "Por favor, cargue una imagen primero.")
            return

        try:
            # Comprimir imagen
            compressed_image = self.compress_image_dct(self.original_image, compression_ratio)
            self.compressed_image = compressed_image

            # Guardar imagen comprimida
            save_name, _ = QFileDialog.getSaveFileName(self, 'Guardar Imagen Comprimida', '', 'Imagenes (*.jpg)')
            if save_name:
                # Guardar la imagen en formato JPG, asegurando compresión
                cv2.imwrite(save_name, compressed_image, [cv2.IMWRITE_JPEG_QUALITY, 90])  # Ajusta la calidad si es necesario
                
                # Calcular y mostrar tasa de compresión
                original_size = os.path.getsize(self.load_image())
                compressed_size = os.path.getsize(save_name)
                
                # Asegurar tasa de compresión positiva
                compression_percentage = max(0, (1 - compressed_size / original_size) * 100)

                QMessageBox.information(self, "Compresión Completa", 
                    f"Imagen comprimida guardada.\n"
                    f"Tasa de compresión: {compression_percentage:.2f}%")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error durante la compresión: {str(e)}")
       
# Ventana principal de la aplicación
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
         # Estilo de la aplicación con paleta de colores azul-gris y fuente Roboto
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F0F4F8;
                font-family: 'Roboto', sans-serif;
            }
            QPushButton {
                background-color: #3B7EA1;
                color: white;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                font-family: 'Roboto', sans-serif;
            }
            QPushButton:hover {
                background-color: #2C6E93;
            }
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #3B7EA1;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Roboto', sans-serif;
            }
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #3B7EA1;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Roboto', sans-serif;
            }
        """)
        
        #Inicializaciones 
        self.texture_analyzer = TextureAnalyzer()
        self.geometric_analyzer = GeometricAnalyzer()
        self.image_enhancer = ImageEnhancer()
        self.super_resolution = SuperResolucion()
        self.uploaded_image = None
        self.last_processed_image = None
        self.setup_ui()
        self.fourier_processor = FourierTransformProcessor()
        self.super_resolution = SuperResolucion() 
        self.image_compression = ImageCompression()
        self.dataset_generator = DatasetGenerator()
      
    def setup_ui(self):
        #Setup 
        self.setWindowTitle("Sistema de Procesamiento de Imágenes")
        self.setGeometry(100, 100, 1400, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Panel izquierdo - Visualización
        left_panel = QVBoxLayout()
        self.original_label = QLabel("Imagen Original")
        self.processed_label = QLabel("Imagen Procesada")
        left_panel.addWidget(self.original_label)
        left_panel.addWidget(self.processed_label)

        # Panel central - Características
        central_panel = QVBoxLayout()
        self.features_text = QTextEdit()
        self.features_text.setReadOnly(True)
        central_panel.addWidget(QLabel("Características Extraídas:"))
        central_panel.addWidget(self.features_text)

        # Panel derecho - Controles (inicializado correctamente como QVBoxLayout)
        right_panel = QVBoxLayout()

        # Lista de botones con iconos y métodos asociados
        buttons = [
            ("Cargar Imagen", "camera_icon.png", self.load_image),  
            ("Filtrado por Fourier", "fourier_icon.png", self.show_fourier_filter_dialog),
            ("Filtrado Direccional", "directional_icon.png", self.apply_directional_filter),
            ("Realce Laplaciano", "laplacian_icon.png", self.apply_laplacian_enhancement),
            ("Mejorar Iluminación", "lighten_icon.png", self.enhance_image),
            ("Reducir Ruido", "noise_icon.png", self.reduce_noise),
            ("Umbral Adaptativo", "threshold_icon.png", self.adaptive_threshold),
            ("Extraer Características", "features_icon.png", self.extract_features),
            ("Descargar Imagen Procesada", "download_icon.png", self.download_processed_image),
            ("Comprimir Imagen DCT", "compress_icon.png", self.compress_dct_image),
        ]

        # Crear los botones y agregarles iconos y funcionalidad
        for btn_text, icon_path, btn_method in buttons:
            btn = QPushButton(btn_text)
            btn.setIcon(QIcon(icon_path))  # Establecer el icono si está disponible
            btn.clicked.connect(btn_method)
            right_panel.addWidget(btn)
        
        # Botón para aplicar super resolución
        self.super_res_button = QPushButton('Aplicar Super Resolución', self)
        self.super_res_button.clicked.connect(self.apply_super_resolution)
        right_panel.addWidget(self.super_res_button)
        
        # Botón para generar dataset y entrenar
        self.btnGenerarDataset = QPushButton("Generar Dataset y Entrenar", self)
        self.btnGenerarDataset.clicked.connect(self.preparar_y_entrenar_modelo)
        self.btnGenerarDataset.setStyleSheet("""
            QPushButton {
                background-color: #OF52BA;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
         """)
        self.btnGenerarDataset.setIcon(QIcon('dataset_icon.png')) 
        right_panel.addWidget(self.btnGenerarDataset)
        
        # Agregar los paneles al layout principal
        layout.addLayout(left_panel, stretch=2)    # Panel izquierdo con mayor peso
        layout.addLayout(central_panel, stretch=1)  # Panel central con menor peso
        layout.addLayout(right_panel, stretch=1) # Panel derecho con el mismo peso que el central
            
    def compress_dct_image(self):
        """Método para comprimir imagen usando DCT"""
        if not hasattr(self, 'uploaded_image') or self.uploaded_image is None:
            QMessageBox.warning(self, "Advertencia", "Por favor, cargue una imagen primero.")
            return

        try:
            # Comprimir imagen
            compressed_image = self.image_compression.compress_image_dct(self.uploaded_image, compression_ratio=0.1)
    
            # Convertir imagen comprimida a QPixmap para mostrar
            compressed_image_rgb = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)
            h, w, ch = compressed_image_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(compressed_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            compressed_pixmap = QPixmap.fromImage(q_img)
    
            # Mostrar imagen comprimida
            self.processed_label.setPixmap(compressed_pixmap)
    
            # Guardar imagen
            save_name, _ = QFileDialog.getSaveFileName(self, 'Guardar Imagen Comprimida', '', 'Imagenes (*.jpg)')
            if save_name:
               cv2.imwrite(save_name, compressed_image)
               # Calcular y mostrar tasa de compresión
               original_size = os.path.getsize(self.original_image_path)
               compressed_size = os.path.getsize(save_name)
               compression_percentage = (1 - compressed_size / original_size) * 100

               QMessageBox.information(self, "Compresión Completa", 
                   f"Imagen comprimida guardada.\n"
                   f"Tasa de compresión: {compression_percentage:.2f}%")
    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error durante la compresión: {str(e)}")
        
    def preparar_y_entrenar_modelo(self):
        # Abrir diálogo para seleccionar directorio de imágenes
        directorio_imagenes = QFileDialog.getExistingDirectory(
            self, 
            "Seleccionar Directorio con Imágenes para Entrenamiento"
        )
        
        if directorio_imagenes:
            try:
                # Mostrar diálogo de progreso
                progreso = QProgressDialog("Generando Dataset...", "Cancelar", 0, 100, self)
                progreso.setWindowModality(Qt.WindowModal)
                progreso.show()
                
                # Generar dataset
                X_train, Y_train = self.super_resolution.dataset_generator.generar_imagenes_desde_directorio(directorio_imagenes)
        
                # Entrenar modelo
                self.super_resolution.train_model(X_train=X_train, Y_train=Y_train, batch_size=16, epochs=10)
        
                 # Notificar al usuario
                QMessageBox.information(self, "Entrenamiento Completado", "El modelo ha sido entrenado exitosamente con el nuevo dataset")
        
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Ocurrió un error durante la preparación del dataset: {str(e)}")
                      
    def apply_super_resolution(self):
        """Aplica super resolución"""
        if hasattr(self, 'uploaded_image') and self.uploaded_image is not None:
            try:
                # Asegurarse de que la imagen esté en formato correcto
                if len(self.uploaded_image.shape) == 2:  # Si es imagen en escala de grises
                    self.uploaded_image = cv2.cvtColor(self.uploaded_image, cv2.COLOR_GRAY2BGR)
            
                # Aplicar super resolución
                enhanced = self.super_resolution.aplicar_super_resolucion(self.uploaded_image)
            
                # Convertir a RGB para mostrar
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
                # Convertir a QPixmap
                h, w, ch = enhanced_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(enhanced_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
            
                # Mostrar imagen mejorada
                self.processed_label.setPixmap(pixmap)
            
                # Guardar imagen mejorada
                save_name, _ = QFileDialog.getSaveFileName(self, 'Guardar Imagen Mejorada', '', 'Imagenes (*.png)')
                if save_name:
                   cv2.imwrite(save_name, enhanced)
            
                self.features_text.append("Super resolución aplicada exitosamente")
        
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo aplicar super resolución: {str(e)}")
        else:
            QMessageBox.warning(self, "Advertencia", "Por favor, cargue una imagen primero.")

    def show_fourier_filter_dialog(self):
        """
        Muestra un cuadro de diálogo para elegir entre filtros de Fourier de paso bajo y paso alto
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Filtrado de Fourier")
        layout = QVBoxLayout()

        low_pass_btn = QPushButton("Filtro Paso Bajo")
        high_pass_btn = QPushButton("Filtro Paso Alto")

        low_pass_btn.clicked.connect(lambda: self.apply_fourier_filter('low'))
        high_pass_btn.clicked.connect(lambda: self.apply_fourier_filter('high'))

        layout.addWidget(low_pass_btn)
        layout.addWidget(high_pass_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def apply_fourier_filter(self, filter_type):
        """
        Aplicar el filtro de Fourier seleccionado a la imagen cargada
        """
        if self.uploaded_image is not None:
            try:
                if filter_type == 'low':
                    filtered = self.fourier_processor.low_pass_filter(self.uploaded_image)
                else:
                    filtered = self.fourier_processor.high_pass_filter(self.uploaded_image)
                
                self.last_processed_image = filtered
                self.display_image(filtered, self.processed_label)
                self.features_text.append(f"Filtro Fourier {'Paso Bajo' if filter_type == 'low' else 'Paso Alto'} aplicado")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo aplicar filtro de Fourier: {str(e)}")

    def apply_directional_filter(self):
        """
        Aplicar filtro direccional de Fourier
        """
        if self.uploaded_image is not None:
            try:
                filtered = self.fourier_processor.directional_filter(self.uploaded_image)
                self.last_processed_image = filtered
                self.display_image(filtered, self.processed_label)
                self.features_text.append("Filtrado Direccional aplicado")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo aplicar filtrado direccional: {str(e)}")

    def apply_laplacian_enhancement(self):
        """
        Aplicar mejora laplaciana a la imagen cargada
        """
        if self.uploaded_image is not None:
            try:
                enhanced = self.fourier_processor.laplacian_enhancement(self.uploaded_image)
                self.last_processed_image = enhanced
                self.display_image(enhanced, self.processed_label)
                self.features_text.append("Realce Laplaciano aplicado")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo aplicar realce laplaciano: {str(e)}")
                
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "", "Images (*.png *.jpg *.jpeg)"
        )

        if file_name:
            self.uploaded_image = cv2.imread(file_name)
            self.display_image(self.uploaded_image, self.original_label)
            self.features_text.clear()
            self.features_text.append(f"Imagen cargada: {file_name}")
            self.original_image_path = file_name
          
    def download_processed_image(self):
        if self.last_processed_image is not None:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Guardar Imagen Procesada", "", "Imagen PNG (*.png);;Imagen JPEG (*.jpg)")
            
            if file_name:
                cv2.imwrite(file_name, self.last_processed_image)
                QMessageBox.information(self, "Éxito", "Imagen guardada correctamente")
        else:
            QMessageBox.warning(self, "Error", "No hay imagen procesada para descargar")

    def enhance_image(self):
        if self.uploaded_image is not None:
            enhanced = self.image_enhancer.correct_illumination(self.uploaded_image)
            self.last_processed_image = enhanced
            self.display_image(enhanced, self.processed_label)
            self.features_text.append("Iluminación corregida")

    def reduce_noise(self):
        if self.uploaded_image is not None:
            try:
                denoised = NoiseReduction.reduce_noise(self.uploaded_image)
                self.last_processed_image = denoised
                self.display_image(denoised, self.processed_label)
                self.features_text.append("Filtro Laplaciano y Reducción de Ruido")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo reducir el ruido: {str(e)}")

    def adaptive_threshold(self):
        if self.uploaded_image is not None:
            thresh = self.image_enhancer.adaptive_threshold(self.uploaded_image)
            self.last_processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            self.display_image(self.last_processed_image, self.processed_label)
            self.features_text.append("Umbral adaptativo aplicado")

    def extract_features(self):
        if self.uploaded_image is not None:
            try:
                # Características de textura
                texture_features = self.texture_analyzer.compute_glcm_features(self.uploaded_image)
                
                # Características geométricas
                geometric_features, contours = self.geometric_analyzer.extract_geometric_features(self.uploaded_image)

                # Mostrar características
                self.features_text.clear()
                self.features_text.append("Características de Textura:")
                for name, value in texture_features.items():
                    self.features_text.append(f"{name}: {value.mean():.4f}")

                self.features_text.append("\nCaracterísticas Geométricas:")
                for i, feat in enumerate(geometric_features):
                    self.features_text.append(f"\nObjeto {i + 1}:")
                    self.features_text.append(f"Área: {feat['area']:.2f}")
                    self.features_text.append(f"Perímetro: {feat['perimeter']:.2f}")
                    self.features_text.append(f"Circularidad: {feat['circularity']:.4f}")
                    self.features_text.append(f"Relación de Aspecto: {feat['aspect_ratio']:.4f}")

                # Dibujar contornos
                result = self.uploaded_image.copy()
                cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
                self.last_processed_image = result
                self.display_image(result, self.processed_label)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudieron extraer características: {str(e)}")
                                      
    def display_image(self, image, label):
        # Redimensionar la imagen para que quepa en el label manteniendo la proporción
        label_width = label.width()
        label_height = label.height()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # Calcular el factor de escala para ajustar al label
        scale = min(label_width / w, label_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensionar la imagen
        resized_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        bytes_per_line = ch * new_w
        qt_image = QImage(resized_image.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)      
        
if __name__ == '__main__':     
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)     
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())   