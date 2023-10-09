import cv2
import mediapipe as mp
import math
import time
from screeninfo import get_monitors

# Realizando la videocaptura
cap = cv2.VideoCapture(0)

# Obtener el tamaño de la pantalla del monitor principal
screen_info = get_monitors()[0]
screen_width = screen_info.width
screen_height = screen_info.height

# Configurar el tamaño de la ventana de la webcam
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow("Frame", screen_width, screen_height)
# Variable de conteo
parpadeo = False
conteo = 0
tiempo = 0
inicio = 0
final = 0
conteo_sue = 0
muestra = 0

# Muestra de función de dibujo
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)  # Configuración del dibujo
mpMallaFacial = mp.solutions.face_mesh  # Llamando la función
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)  # Creando el objeto

# Creando el bucle principal
while True:
    ret, frame = cap.read()
    # Correccion de color
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Observamos los resultados
    resultados = MallaFacial.process(frameRGB)

    # Creando lineas para almacenar resultados
    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if resultados.multi_face_landmarks:  # Si se detecta un rostro
        for rostros in resultados.multi_face_landmarks:  # Muestra el rostro detectado
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

            # Extraer los puntos del rostro detectado
            for id, puntos in enumerate(rostros.landmark):
                # print(puntos)#Nos entrega una proporcion
                al, an, c = frame.shape
                x, y = int(puntos.x * an), int(puntos.y * al)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])
                if len(lista) == 468:
                    # Ojo derecho
                    x1, y1 = lista[145][1:]
                    x2, y2 = lista[159][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    longitud = math.hypot(x2 - x1, y2 - y1)
                    # Ojo izquierdo
                    x3, y3 = lista[374][1:]
                    x4, y4 = lista[386][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 - y3)

                    # Conteo de parpadeos
                    cv2.putText(frame, f'Parpadeos:{int(conteo)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(frame, f'MicroSueños:{int(conteo_sue)}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(frame, f'Duracion:{str(muestra)}', (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    if longitud <= 10 and longitud2 <= 10 and parpadeo == False:  # Parpadeo
                        conteo = conteo + 1
                        parpadeo = True
                        inicio = time.time()
                    elif longitud > 10 and longitud2 > 10 and parpadeo == True:
                        parpadeo = False
                        final = time.time()

                        # Seguridad parpadeo

                        # Temporizador
                        tiempo = round(final - inicio, 0)

                        # Contador de Micro Sueños
                        if tiempo >= 3:
                            conteo_sue = conteo_sue + 1
                            muestra = tiempo
                            inicio = 0
                            final = 0

    # Mostrar el marco resultante
    cv2.imshow("Frame", frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()






