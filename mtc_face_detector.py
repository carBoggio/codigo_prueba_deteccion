import cv2
import numpy as np
from mtcnn import MTCNN
import time

class LiveFaceDetector:
    """
    Detector facial en tiempo real usando OpenCV para capturar frames
    y MTCNN para la detección facial
    """
    def __init__(self):
        """
        Inicializa el detector MTCNN
        """
        # Inicializar detector MTCNN
        self.detector = MTCNN()
        print("Detector MTCNN inicializado")
        
        # Variables para la cámara
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        
    def start_camera(self, camera_id=0, width=640, height=480):
        """
        Inicia la captura de la webcam
        
        Args:
            camera_id: ID de la cámara (0 por defecto)
            width: Ancho deseado del frame
            height: Alto deseado del frame
        
        Returns:
            bool: True si la cámara se inició correctamente
        """
        self.cap = cv2.VideoCapture(camera_id)
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.frame_width = width
        self.frame_height = height
        
        # Verificar si la cámara se abrió correctamente
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return False
            
        print(f"Cámara iniciada (ID: {camera_id}, Resolución: {width}x{height})")
        return True
    
    def release_camera(self):
        """
        Libera la cámara y cierra todas las ventanas
        """
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Cámara liberada")
    
    def detect_faces_in_frame(self, frame):
        """
        Detecta rostros en un frame usando MTCNN
        
        Args:
            frame: Frame de video como array numpy
        
        Returns:
            frame_with_boxes: Frame con cajas delimitadoras y puntos faciales
            face_detections: Lista de detecciones con información completa
        """
        # Hacer una copia del frame para dibujar
        frame_with_boxes = frame.copy()
        
        # Calcular FPS
        self.frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.last_time
        
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.last_time = current_time
        
        # Detectar rostros
        try:
            # MTCNN espera RGB, OpenCV usa BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Iniciar temporizador para medir el tiempo de detección
            start_time = time.time()
            
            # Ejecutar la detección facial con MTCNN
            face_detections = self.detector.detect_faces(rgb_frame)
            
            # Calcular el tiempo que tomó la detección
            detection_time = (time.time() - start_time) * 1000  # Convertir a milisegundos
            print(f"Tiempo de detección MTCNN: {detection_time:.2f} ms")
            
            # Dibujar información de FPS
            cv2.putText(frame_with_boxes, f"FPS: {self.fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Dibujar número de rostros detectados
            cv2.putText(frame_with_boxes, f"Rostros: {len(face_detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar tiempo de detección en el frame
            cv2.putText(frame_with_boxes, f"Tiempo: {detection_time:.1f} ms", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Procesar cada detección
            for detection in face_detections:
                # Obtener coordenadas del cuadro delimitador
                x, y, width, height = detection['box']
                confidence = detection['confidence']
                
                # Dibujar rectángulo alrededor del rostro
                cv2.rectangle(frame_with_boxes, (x, y), (x+width, y+height), 
                             (0, 255, 0), 2)
                
                # Mostrar confianza
                cv2.putText(frame_with_boxes, f"{confidence:.2f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Dibujar puntos clave faciales (ojos, nariz, boca)
                keypoints = detection['keypoints']
                for point_name, (px, py) in keypoints.items():
                    cv2.circle(frame_with_boxes, (px, py), 2, (0, 0, 255), 2)
            
            return frame_with_boxes, face_detections
            
        except Exception as e:
            print(f"Error en la detección: {e}")
            return frame, []
    
    def extract_aligned_faces(self, frame, detections, face_size=(112, 112)):
        """
        Extrae rostros alineados de las detecciones para su uso posterior
        
        Args:
            frame: Frame original
            detections: Detecciones de MTCNN
            face_size: Tamaño deseado para los rostros extraídos
            
        Returns:
            list: Lista de rostros extraídos y alineados
        """
        aligned_faces = []
        
        for detection in detections:
            # Obtener coordenadas
            x, y, width, height = detection['box']
            
            # Obtener puntos clave
            keypoints = detection['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            
            # Extraer rostro
            face = frame[y:y+height, x:x+width].copy()
            
            # Verificar que la cara es válida
            if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                continue
                
            # Alinear rostro usando puntos de los ojos
            # (esto es simplificado, una alineación completa usaría transformaciones afines)
            try:
                # Redimensionar rostro al tamaño deseado
                face = cv2.resize(face, face_size)
                
                # Añadir a la lista junto con su confianza
                aligned_faces.append({
                    'face': face,
                    'confidence': detection['confidence'],
                    'box': detection['box'],
                    'keypoints': detection['keypoints']
                })
            except Exception as e:
                print(f"Error al procesar rostro: {e}")
                
        return aligned_faces
    
    def run_detection_loop(self):
        """
        Ejecuta el bucle principal de detección en tiempo real
        """
        if self.cap is None or not self.cap.isOpened():
            print("Error: La cámara no está configurada correctamente.")
            return
            
        print("Iniciando detección facial en tiempo real. Presiona 'q' para salir.")
        
        try:
            while True:
                # Capturar frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error al leer frame de la cámara.")
                    break
                    
                # Detectar rostros
                frame_with_detections, detections = self.detect_faces_in_frame(frame)
                
                # Extraer rostros alineados (opcional)
                if detections:
                    aligned_faces = self.extract_aligned_faces(frame, detections)
                    
                    # Mostrar rostros extraídos en ventanas separadas (máximo 5)
                    for i, face_data in enumerate(aligned_faces[:5]):
                        face_img = face_data['face']
                        conf = face_data['confidence']
                        
                        # Añadir confianza al rostro extraído
                        face_with_text = face_img.copy()
                        cv2.putText(face_with_text, f"Conf: {conf:.2f}", (5, 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        cv2.imshow(f"Rostro {i+1}", face_with_text)
                
                # Mostrar frame con detecciones
                cv2.imshow("Detección Facial MTCNN", frame_with_detections)
                
                # Salir con la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Detección interrumpida por el usuario")
        except Exception as e:
            print(f"Error en el bucle de detección: {e}")
        finally:
            self.release_camera()
            

# Ejecutar detección si este script es el principal
if __name__ == "__main__":
    # Crear detector
    detector = LiveFaceDetector()
    
    # Iniciar cámara (puedes cambiar el ID de la cámara si tienes múltiples)
    if detector.start_camera(camera_id=0, width=640, height=480):
        # Ejecutar bucle de detección
        detector.run_detection_loop()
    else:
        print("No se pudo iniciar la cámara. Verifique la conexión o los permisos.")