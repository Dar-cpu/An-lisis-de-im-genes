import os
import cv2
import numpy as np
import logging
from typing import Tuple, List , Optional
import tensorflow as tf

class DatasetGenerator:
    def __init__(self, base_dir: str = 'dataset_superresolution'):
        """
        Inicializa el generador de dataset para super resolución con mejor manejo de errores
        
        Args:
            base_dir (str): Directorio base para almacenar imágenes
        """
        self.base_dir = base_dir
        self.hr_dir = os.path.join(base_dir, 'alta_resolucion')
        self.lr_dir = os.path.join(base_dir, 'baja_resolucion')
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Crear directorios si no existen
        os.makedirs(self.hr_dir, exist_ok=True)
        os.makedirs(self.lr_dir, exist_ok=True)

    def _redimensionar_imagen(self, imagen: np.ndarray, scale_factor: float) -> Optional[np.ndarray]:
        """
        Redimensiona la imagen manteniendo la proporción con validaciones adicionales
        
        Args:
            imagen (np.ndarray): Imagen original
            scale_factor (float): Factor de escala para reducción
        
        Returns:
            Optional[np.ndarray]: Imagen redimensionada o None si hay error
        """
        try:
            # Validar que la imagen no esté vacía
            if imagen is None or imagen.size == 0:
                self.logger.error("Imagen vacía o inválida")
                return None

            # Validar dimensiones mínimas
            if imagen.shape[0] < 10 or imagen.shape[1] < 10:
                self.logger.warning("Imagen demasiado pequeña para redimensionar")
                return None
            
            if imagen.shape[0] < 10 or imagen.shape[1] < 10:
                self.logger.warning("Imagen demasiado pequeña para redimensionar.")
                return None

            nuevo_ancho = max(1, int(imagen.shape[1] * scale_factor))
            nuevo_alto = max(1, int(imagen.shape[0] * scale_factor))
            
            # Usar interpolación INTER_LINEAR para mejor calidad
            imagen_redimensionada = cv2.resize(
                imagen, 
                (nuevo_ancho, nuevo_alto), 
                interpolation=cv2.INTER_LINEAR
            )
            
            return imagen_redimensionada
        
        except Exception as e:
            self.logger.error(f"Error al redimensionar imagen: {e}")
            return None

    def generar_imagenes_desde_directorio(self, 
                                          directorio_origen: str, 
                                          scale_factors: List[float] = [0.5, 0.75],
                                          extensiones: Tuple[str, ...] = ('.jpg', '.png', '.jpeg', '.bmp')) -> int:
        """
        Genera imágenes de alta y baja resolución desde un directorio de origen
        
        Args:
            directorio_origen (str): Ruta del directorio con imágenes originales
            scale_factors (List[float]): Factores de reducción para imágenes LR
            extensiones (Tuple[str]): Extensiones de imagen válidas
        
        Returns:
            int: Número de imágenes procesadas exitosamente
        """
        # Validar directorio de origen
        if not os.path.exists(directorio_origen):
            self.logger.error(f"Directorio de origen no existe: {directorio_origen}")
            return 0
        
        # Contador para nombrar archivos
        contador = 0
        
        # Recorrer archivos en directorio origen
        for nombre_archivo in os.listdir(directorio_origen):
            # Validar extensión
            if any(nombre_archivo.lower().endswith(ext) for ext in extensiones):
                ruta_imagen = os.path.join(directorio_origen, nombre_archivo)
                
                try:
                    # Cargar imagen con color
                    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
                    
                    # Verificar que sea una imagen válida
                    if imagen is None or imagen.size == 0:
                        self.logger.warning(f"No se pudo leer la imagen: {ruta_imagen}")
                        continue
                    
                    # Generar nombre base para la imagen
                    nombre_base = f'imagen_{contador:04d}'
                    
                    # Copiar imagen original como HR - convertir a RGB
                    hr_path = os.path.join(self.hr_dir, f'{nombre_base}_hr.png')
                    cv2.imwrite(hr_path, imagen, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    
                    # Generar versiones LR con diferentes escalas
                    for factor in scale_factors:
                        lr_imagen = self._redimensionar_imagen(imagen, factor)
                        
                        if lr_imagen is not None:
                            lr_path = os.path.join(self.lr_dir, f'{nombre_base}_lr_{factor:.2f}.png')
                            cv2.imwrite(lr_path, lr_imagen, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    
                    contador += 1
                    
                except Exception as e:
                    self.logger.error(f"Error procesando {nombre_archivo}: {e}")
        
        self.logger.info(f"Procesadas {contador} imágenes exitosamente")
        return contador

    def obtener_datos_entrenamiento(self,   
                                    image_size: Tuple[int, int] = (128, 128), 
                                    normalizar: bool = True,
                                    convertir_rgb: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga las imágenes de alta y baja resolución para entrenamiento con mejoras
        
        Args:
            image_size (Tuple[int, int]): Tamaño de redimensión de imágenes
            normalizar (bool): Normalizar valores de pixel entre 0 y 1
            convertir_rgb (bool): Convertir imágenes de BGR a RGB
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: X_train (LR), Y_train (HR)
        """
        X_train, Y_train = [], []
        
        try:
            # Listar archivos LR
            archivos_lr = sorted(os.listdir(self.lr_dir))
        
            if not archivos_lr:
               self.logger.error("No se encontraron imágenes de baja resolución")
               return np.array([]), np.array([])
        
            for archivo_lr in archivos_lr:
                try:
                    # Buscar imagen HR correspondiente
                    nombre_base = archivo_lr.split('_lr_')[0]
                    archivos_hr = [f for f in os.listdir(self.hr_dir) if f.startswith(nombre_base) and f.endswith('_hr.png')]
                
                    if not archivos_hr:
                        self.logger.warning(f"No se encontró imagen HR para: {archivo_lr}")
                        continue
                
                    lr_path = os.path.join(self.lr_dir, archivo_lr)
                    hr_path = os.path.join(self.hr_dir, archivos_hr[0])
                
                    # Cargar imágenes con validaciones
                    lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
                    hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
                
                    if lr_img is None or hr_img is None:
                        self.logger.error(f"Error al cargar imágenes: LR={lr_path}, HR={hr_path}")
                        continue
                
                    # Convertir de BGR a RGB si es necesario
                    if convertir_rgb:
                        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                
                    # Redimensionar con interpolación consistente
                    lr_img = cv2.resize(lr_img, image_size, interpolation=cv2.INTER_LINEAR)
                    hr_img = cv2.resize(hr_img, image_size, interpolation=cv2.INTER_LINEAR)
                
                    #Normalizar si es necesario
                    if normalizar:
                        lr_img = lr_img.astype(np.float32) / 255.0
                        hr_img = hr_img.astype(np.float32) / 255.0
                
                    X_train.append(lr_img)
                    Y_train.append(hr_img)
                
                except Exception as e:
                    self.logger.error(f"Error procesando par de imágenes {archivo_lr}: {e}")
        
            # Convertir a numpy arrays
            if not X_train or not Y_train:
                self.logger.error("No se generaron datos de entrenamiento")
                return np.array([]), np.array([])
            
            print(f"Archivos LR encontrados: {archivos_lr}")
            print(f"Directorio LR: {self.lr_dir}")
            print(f"Directorio HR: {self.hr_dir}")
        
            return np.array(X_train), np.array(Y_train)
    
        except Exception as e:
            self.logger.error(f"Error general en obtener_datos_entrenamiento: {e}")
            return np.array([]), np.array([])