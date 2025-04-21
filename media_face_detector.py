import cv2
import mediapipe as mp
import numpy as np
import time

class FaceDetector:
    """
    Clase para la detección facial usando OpenCV y MediaPipe para detección
    """
    def __init__(self, min_detection_confidence=0.5, model_selection=1):
        """
        Inicializa el detector facial con MediaPipe
        
        Args:
            min_detection_confidence: Umbral de confianza mínima para detecciones
            model_selection: 0 para detección a corta distancia, 1 para detección a larga distancia
        """
        # Inicializar componentes de MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configurar el detector facial
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
        
        # Estado del procesamiento
        self.is_running = False
        self.image = None
        
        print("Detector facial MediaPipe inicializado")
    
    def load_image(self, image_path):
        """
        Carga una imagen desde un archivo
        
        Args:
            image_path: Ruta a la imagen que se desea cargar
            
        Returns:
            bool: True si la imagen se cargó correctamente, False en caso contrario
        """
        # Cargar la imagen con OpenCV
        self.image = cv2.imread(image_path)
        
        # Verificar si la imagen se cargó correctamente
        if self.image is None:
            print(f"Error: No se pudo cargar la imagen desde {image_path}.")
            return False
        
        self.is_running = True
        print(f"Imagen cargada correctamente. Dimensiones: {self.image.shape[1]}x{self.image.shape[0]}")
        return True
    
    def stop_processing(self):
        """
        Detiene el procesamiento y libera recursos
        """
        self.is_running = False
        cv2.destroyAllWindows()
        print("Procesamiento detenido")
    
    def detect_faces(self, image):
        """
        Detecta rostros en una imagen usando MediaPipe
        
        Args:
            image: Imagen en formato numpy array (BGR)
            
        Returns:
            processed_image: Imagen con anotaciones de detección
            detections: Lista de detecciones faciales
        """
        # Convertir la imagen de BGR a RGB (MediaPipe requiere RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Iniciar temporizador
        start_time = time.time()
        
        # Procesar la imagen con MediaPipe
        results = self.face_detection.process(image_rgb)
        
        # Calcular tiempo de procesamiento
        process_time = (time.time() - start_time) * 1000  # Convertir a milisegundos
        print(f"Tiempo de detección: {process_time:.2f} ms")
        
        # Crear una copia de la imagen para dibujar encima
        processed_image = image.copy()
        
        detections = []
        
        # Verificar si se encontraron rostros
        if results.detections:
            for detection in results.detections:
                # Obtener cuadro delimitador normalizado (en formato [0,1])
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                
                # Convertir a coordenadas de píxeles
                x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Dibujar rectángulo alrededor del rostro
                cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Dibujar puntos clave faciales (ojos, nariz, boca, orejas)
                self.mp_drawing.draw_detection(processed_image, detection)
                
                # Mostrar puntuación de confianza
                confidence = detection.score[0]
                cv2.putText(processed_image, f"Conf: {confidence:.2f}", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Extraer región facial para procesamiento posterior
                try:
                    face_roi = image[y:y+h, x:x+w]
                    # Asegurarse de que la ROI es válida
                    if face_roi.size > 0:
                        # Guardar información de detección
                        detection_info = {
                            'bbox': (x, y, w, h),
                            'confidence': float(confidence),
                            'landmarks': [],  # Extraer puntos clave si se necesitan
                            'face_roi': face_roi
                        }
                        
                        # Extraer puntos clave faciales normalizados
                        for i in range(6):  # 6 puntos clave en MediaPipe Face Detection
                            try:
                                landmark = detection.location_data.relative_keypoints[i]
                                px, py = int(landmark.x * iw), int(landmark.y * ih)
                                detection_info['landmarks'].append((px, py))
                            except:
                                pass
                                
                        detections.append(detection_info)
                except Exception as e:
                    print(f"Error al extraer región facial: {e}")
        
        return processed_image, detections
    
    def process_static_image(self, process_detections_callback=None):
        """
        Procesa una imagen estática y muestra el resultado indefinidamente
        
        Args:
            process_detections_callback: Función opcional para procesar detecciones
                                       Debe aceptar (frame, detections) como argumentos
        """
        if not self.is_running or self.image is None:
            print("Error: No hay imagen cargada para procesar")
            return
        
        try:
            # Usar la imagen cargada como frame
            frame = self.image.copy()
            
            # Detectar rostros
            processed_frame, detections = self.detect_faces(frame)
            
            # Mostrar información sobre detecciones
            cv2.putText(processed_frame, f"Rostros: {len(detections)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Si hay una función de callback, llamarla con el frame procesado y detecciones
            if process_detections_callback is not None:
                process_detections_callback(processed_frame, detections)
            
            # Mostrar resultado indefinidamente hasta que se presione 'q'
            print("Mostrando resultado. Presiona 'q' para salir.")
            while self.is_running:
                cv2.imshow('Detección Facial MediaPipe', processed_frame)
                
                # Salir si se presiona la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error en el procesamiento de la imagen: {e}")
        
        finally:
            # Liberar recursos
            cv2.destroyAllWindows()
            self.is_running = False
    
    def extract_face_for_recognition(self, image, detection, target_size=(112, 112)):
        """
        Extrae y preprocesa una cara detectada para reconocimiento facial
        
        Args:
            image: Imagen original
            detection: Información de detección facial
            target_size: Tamaño deseado para la imagen facial
            
        Returns:
            Face image normalized and resized for recognition
        """
        x, y, w, h = detection['bbox']
        
        # Extraer región facial
        face_roi = image[y:y+h, x:x+w]
        
        # Verificar que la ROI es válida
        if face_roi.size == 0:
            return None
        
        # Redimensionar para reconocimiento facial
        face_roi = cv2.resize(face_roi, target_size)
        
        return face_roi


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar detector
    detector = FaceDetector(min_detection_confidence=0.5)
    
    # Ruta a la imagen que quieres procesar
    image_path = "./cara_ejemplo.jpg"  # Cambia esto a la ruta de tu imagen
    
    # Cargar imagen
    if detector.load_image(image_path):
        # Definir callback para procesar detecciones (opcional)
        def process_detections(frame, detections):
            # Aquí puedes implementar lógica adicional con las detecciones
            for i, det in enumerate(detections):
                confidence = det['confidence']
                print(f"Rostro {i+1}: Confianza = {confidence:.2f}")
                
                # Extraer cara para reconocimiento
                face = detector.extract_face_for_recognition(frame, det)
                if face is not None:
                    # Mostrar cara extraída en una ventana separada
                    cv2.imshow(f"Rostro {i+1}", face)
        
        # Procesar imagen estática
        detector.process_static_image(process_detections_callback=process_detections)
    else:
        print(f"No se pudo cargar la imagen: {image_path}")