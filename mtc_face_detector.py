import cv2
import numpy as np
from mtcnn import MTCNN
import time

class StaticFaceDetector:
    """
    Detector facial usando MTCNN para imágenes estáticas
    """
    def __init__(self):
        """
        Inicializa el detector MTCNN
        """
        # Inicializar detector MTCNN
        self.detector = MTCNN()
        print("Detector MTCNN inicializado")
        
        # Variables para la imagen
        self.image = None
        self.is_running = False
        
    def load_image(self, image_path):
        """
        Carga una imagen desde el disco
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            bool: True si la imagen se cargó correctamente
        """
        # Cargar imagen con OpenCV
        self.image = cv2.imread(image_path)
        
        # Verificar si la imagen se cargó correctamente
        if self.image is None:
            print(f"Error: No se pudo cargar la imagen desde {image_path}.")
            return False
            
        self.is_running = True
        print(f"Imagen cargada correctamente. Dimensiones: {self.image.shape[1]}x{self.image.shape[0]}")
        return True
    
    def release_resources(self):
        """
        Libera recursos y cierra todas las ventanas
        """
        cv2.destroyAllWindows()
        self.is_running = False
        print("Recursos liberados")
    
    def detect_faces_in_image(self, image):
        """
        Detecta rostros en una imagen usando MTCNN
        
        Args:
            image: Imagen como array numpy
        
        Returns:
            image_with_boxes: Imagen con cajas delimitadoras y puntos faciales
            face_detections: Lista de detecciones con información completa
        """
        # Hacer una copia de la imagen para dibujar
        image_with_boxes = image.copy()
        
        # Detectar rostros
        try:
            # MTCNN espera RGB, OpenCV usa BGR
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Iniciar temporizador para medir el tiempo de detección
            start_time = time.time()
            
            # Ejecutar la detección facial con MTCNN
            face_detections = self.detector.detect_faces(rgb_image)
            
            # Calcular el tiempo que tomó la detección
            detection_time = (time.time() - start_time) * 1000  # Convertir a milisegundos
            print(f"Tiempo de detección MTCNN: {detection_time:.2f} ms")
            
            # Dibujar número de rostros detectados
            cv2.putText(image_with_boxes, f"Rostros: {len(face_detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar tiempo de detección en la imagen
            cv2.putText(image_with_boxes, f"Tiempo: {detection_time:.1f} ms", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Procesar cada detección
            for detection in face_detections:
                # Obtener coordenadas del cuadro delimitador
                x, y, width, height = detection['box']
                confidence = detection['confidence']
                
                # Dibujar rectángulo alrededor del rostro
                cv2.rectangle(image_with_boxes, (x, y), (x+width, y+height), 
                             (0, 255, 0), 2)
                
                # Mostrar confianza
                cv2.putText(image_with_boxes, f"{confidence:.2f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Dibujar puntos clave faciales (ojos, nariz, boca)
                keypoints = detection['keypoints']
                for point_name, (px, py) in keypoints.items():
                    cv2.circle(image_with_boxes, (px, py), 2, (0, 0, 255), 2)
            
            return image_with_boxes, face_detections
            
        except Exception as e:
            print(f"Error en la detección: {e}")
            return image, []
    
    def extract_aligned_faces(self, image, detections, face_size=(112, 112)):
        """
        Extrae rostros alineados de las detecciones para su uso posterior
        
        Args:
            image: Imagen original
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
            face = image[y:y+height, x:x+width].copy()
            
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
    
    def process_static_image(self):
        """
        Procesa una imagen estática y muestra los resultados
        """
        if not self.is_running or self.image is None:
            print("Error: No hay imagen cargada para procesar.")
            return
            
        print("Procesando imagen. Presiona 'q' para salir.")
        
        try:
            # Usar la imagen cargada
            image = self.image.copy()
            
            # Detectar rostros
            image_with_detections, detections = self.detect_faces_in_image(image)
            
            # Extraer rostros alineados
            if detections:
                aligned_faces = self.extract_aligned_faces(image, detections)
                
                # Mostrar rostros extraídos en ventanas separadas (máximo 5)
                for i, face_data in enumerate(aligned_faces[:5]):
                    face_img = face_data['face']
                    conf = face_data['confidence']
                    
                    # Añadir confianza al rostro extraído
                    face_with_text = face_img.copy()
                    cv2.putText(face_with_text, f"Conf: {conf:.2f}", (5, 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imshow(f"Rostro {i+1}", face_with_text)
            
            # Mostrar imagen con detecciones indefinidamente hasta presionar 'q'
            while self.is_running:
                cv2.imshow("Detección Facial MTCNN", image_with_detections)
                
                # Salir con la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error en el procesamiento de la imagen: {e}")
        finally:
            self.release_resources()
            

# Ejecutar detección si este script es el principal
if __name__ == "__main__":
    # Crear detector
    detector = StaticFaceDetector()
    
    # Ruta a la imagen que quieres procesar
    image_path = "./cara_ejemplo.jpg"  # Cambia esto a la ruta de tu imagen
    
    # Cargar imagen
    if detector.load_image(image_path):
        # Procesar la imagen estática
        detector.process_static_image()
    else:
        print(f"No se pudo cargar la imagen: {image_path}")