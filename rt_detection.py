import cv2
import numpy as np
import time
from retina_face import RetinaFace

# Cargar imagen
image_path = "./cara_ejemplo.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: No se pudo cargar la imagen desde {image_path}")
    exit()

# Detectar rostros
start_time = time.time()
faces = RetinaFace.detect_faces(image)
detection_time = (time.time() - start_time) * 1000

print(f"Tiempo de detección: {detection_time:.2f} ms")
print(f"Rostros detectados: {len(faces) if isinstance(faces, dict) else 0}")

# Dibujar las detecciones
if isinstance(faces, dict):
    for face_key in faces:
        face = faces[face_key]
        
        # Dibujar rectángulo
        facial_area = face["facial_area"]
        cv2.rectangle(image, 
                      (facial_area[0], facial_area[1]), 
                      (facial_area[2], facial_area[3]), 
                      (0, 255, 0), 2)
        
        # Dibujar puntos faciales
        landmarks = face["landmarks"]
        for landmark_key, landmark_value in landmarks.items():
            point = (int(landmark_value[0]), int(landmark_value[1]))
            cv2.circle(image, point, 2, (0, 0, 255), 2)
        
        # Mostrar confianza
        confidence = face["score"] if "score" in face else face["confidence"] if "confidence" in face else 0
        cv2.putText(image, f"{confidence:.2f}", 
                   (facial_area[0], facial_area[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen con las detecciones indefinidamente hasta que se presione 'q'
print("Mostrando resultados. Presiona 'q' para salir.")
while True:
    cv2.imshow("RetinaFace", image)
    
    # Esperar por tecla presionada (1ms) y verificar si es 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar todas las ventanas cuando el usuario decida salir
cv2.destroyAllWindows()