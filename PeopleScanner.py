import cv2

# Carrega os classificadores pré-treinados para detecção de rostos e olhos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inicia a captura de vídeo
cap = cv2.VideoCapture(0)

def avistei_olhos():
    print('OLHOS')

def avistei_rosto():
    print('ROSTO')

while True:
    # Captura frame a frame
    ret, frame = cap.read()

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Para cada rosto detectado, detecta olhos
    for (x, y, w, h) in faces:
        # Chama a função de detecção de rosto
        avistei_rosto()

        # Desenha um retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Região de interesse (ROI) para os olhos
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detecta olhos na ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Chama a função de detecção de olhos
            avistei_olhos()

            # Desenha um retângulo ao redor de cada olho
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    # Mostra o frame resultante
    cv2.imshow('Face and Eye Detection', frame)

    # Se a tecla 'q' for pressionada, encerra o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos e fecha a janela
cap.release()
cv2.destroyAllWindows()
