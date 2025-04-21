import cv2
import mediapipe as mp  # Esto debería importar la biblioteca, no tu archivo
import numpy as np
import time

class FaceDetector:
    """
    Clase para la detección facial usando OpenCV para capturar frames y MediaPipe para detección
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
        
        # Estado de la cámara
        self.cap = None
        self.is_running = False
        
        print("Detector facial MediaPipe inicializado")
    
    def start_camera(self, camera_id=0, width=640, height=480):
        """
        Inicia la captura de cámara con OpenCV
        
        Args:
            camera_id: ID de la cámara (0 por defecto para la webcam principal)
            width: Ancho deseado para la captura
            height: Alto deseado para la captura
            
        Returns:
            bool: True si la cámara se inició correctamente, False en caso contrario
        """
        # Si ya hay una cámara activa, detenerla primero
        if self.is_running:
            self.stop_camera()
        
        # Iniciar captura de cámara
        self.cap = cv2.VideoCapture(camera_id)
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verificar si la cámara se abrió correctamente
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return False
        
        self.is_running = True
        print(f"Cámara iniciada (ID: {camera_id}, Resolución: {width}x{height})")
        return True
    
    def stop_camera(self):
        """
        Detiene la captura de cámara
        """
        if self.cap is not None and self.is_running:
            self.cap.release()
            self.is_running = False
            print("Cámara detenida")
    
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
    
    def process_webcam_feed(self, process_frame_callback=None):
        """
        Procesa continuamente el feed de la webcam y aplica detección facial
        
        Args:
            process_frame_callback: Función opcional para procesar cada frame y detecciones
                                   Debe aceptar (frame, detections) como argumentos
        """
        if not self.is_running or self.cap is None:
            print("Error: La cámara no está activa")
            return
        
        try:
            while self.is_running:
                # Capturar frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: No se pudo leer el frame")
                    break
                
                # Detectar rostros
                processed_frame, detections = self.detect_faces(frame)
                
                # Mostrar información sobre detecciones
                cv2.putText(processed_frame, f"Rostros: {len(detections)}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Si hay una función de callback, llamarla con el frame procesado y detecciones
                if process_frame_callback is not None:
                    process_frame_callback(processed_frame, detections)
                
                # Mostrar resultado
                cv2.imshow('Detección Facial MediaPipe', processed_frame)
                
                # Salir si se presiona la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error en el procesamiento del video: {e}")
        
        finally:
            # Liberar recursos
            cv2.destroyAllWindows()
            self.stop_camera()
    
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
        
        # Convertir a RGB si el modelo lo requiere
        # face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        return face_roi


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar detector
    detector = FaceDetector(min_detection_confidence=0.5)
    
    # Iniciar cámara
    if detector.start_camera(width=640, height=480):
        # Definir callback para procesar frames (opcional)
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
        
        # Procesar feed de webcam
        detector.process_webcam_feed(process_frame_callback=process_detections)