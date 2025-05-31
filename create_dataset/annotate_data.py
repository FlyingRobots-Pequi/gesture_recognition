import time, sys, os, cv2
import pandas as pd
import mediapipe as mp

def get_class_number():
    while True:
        try:
            class_num = int(input("Digite o número da classe que deseja anotar (1-12): "))
            if 1 <= class_num <= 12:
                return str(class_num)
            else:
                print("Por favor, digite um número entre 1 e 12.")
        except ValueError:
            print("Por favor, digite um número válido.")

# Obtém a classe a ser anotada
current_class = get_class_number()

df_landmarks = pd.DataFrame()
cwd = os.getcwd()
base_images_directory = 'data/'
base_images_directory = os.path.join(cwd, base_images_directory)

# Cria diretório específico para a classe
images_directory = os.path.join(base_images_directory, f'class_{current_class}')
os.makedirs(images_directory, exist_ok=True)

# Gera um arquivo CSV para todas as classes
csv_file_path = 'datapose.csv'

# Verifica se o arquivo CSV já existe para adicionar o cabeçalho apenas se for novo
file_exists = os.path.isfile(csv_file_path)

image_counter = len(os.listdir(images_directory))

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Configurações de desenho para a pose
drawing_styles = mp.solutions.drawing_styles
pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()

# Inicializa a captura de vídeo com configurações otimizadas
cap = cv2.VideoCapture(6) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduz a resolução para melhor performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Define FPS para 30

print(f"\nIniciando anotação para a classe {current_class}")

# Variável para controlar o tempo entre capturas
last_capture_time = 0
capture_interval = 0.1

# Tempo de espera inicial (10 segundos)
start_time = time.time()
waiting = True

# Tempo limite de captura (1 minuto)
capture_time_limit = 60 
capture_start_time = None

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, image = cap.read()
    
        # Verifica se o frame foi capturado corretamente
        if not ret:
            print("Erro ao capturar o frame; saindo...")
            sys.exit(0)

        # Espelha a imagem para evitar efeito de imagem invertida
        image = cv2.flip(image, 1)

        # Converte a imagem de BGR para RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processa a imagem e detecta a pose
        results_pose = pose.process(image_rgb)

        # Desenha as marcações da pose se uma pose for detectada.
        if results_pose.pose_landmarks:
            mp_draw.draw_landmarks(
                image,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            # Se ainda estiver no período de espera, mostra a contagem regressiva
            if waiting:
                elapsed_time = time.time() - start_time
                if elapsed_time < 10:
                    countdown = 10 - int(elapsed_time)
                    cv2.putText(image, f"Comecando em: {countdown}", (5, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    waiting = False
                    capture_start_time = time.time()  # Inicia o contador de captura
                    print("Comecando a captura!")

            # Se já passou o tempo de espera, captura normalmente
            elif not waiting:
                # Verifica se já passou o tempo limite de captura
                if time.time() - capture_start_time >= capture_time_limit:
                    print("\nTempo limite de captura atingido (1 minuto)")
                    break

                current_time = time.time()
                if current_time - last_capture_time >= capture_interval:
                    landmarks_data = {}
                    landmark_names = [landmark.name for landmark in mp_pose.PoseLandmark]
                    for i, landmark in enumerate(results_pose.pose_landmarks.landmark):
                        if i not in list(range(0, 11)) + list(range(25, 33)):  # Exclui os índices que você não quer.
                            landmarks_data[f'{landmark_names[i]}.x'] = round(landmark.x, 4)
                            landmarks_data[f'{landmark_names[i]}.y'] = round(landmark.y, 4)
                            landmarks_data[f'{landmark_names[i]}.z'] = round(landmark.z, 4)
                            landmarks_data[f'{landmark_names[i]}.visibility'] = round(landmark.visibility, 4)

                    # Cria um DataFrame com uma única linha usando o dicionário acima.
                    new_row = pd.DataFrame([landmarks_data])
                    # Adiciona a coluna 'Label' com a classe atual.
                    new_row['Label'] = current_class
                    image_path = os.path.join(images_directory, f'image_{image_counter}.png')
                    new_row['Image_Path'] = image_path 
                    cv2.imwrite(image_path, image)

                    # Concatena a nova linha ao DataFrame existente.
                    df_landmarks = pd.concat([df_landmarks, new_row], ignore_index=True)

                    # Adiciona ao CSV existente ou cria um novo
                    new_row.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)
                    file_exists = True  # Após a primeira escrita, o arquivo existe

                    print(f"Anotando classe: {current_class} - Imagem {image_counter}")
                    image_counter += 1
                    last_capture_time = current_time

        # Instruções na janela.
        cv2.putText(image, f"Anotando classe: {current_class}", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"Imagens capturadas: {image_counter}", (5, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Mostra o tempo restante de captura
        if not waiting and capture_start_time is not None:
            remaining_time = capture_time_limit - (time.time() - capture_start_time)
            if remaining_time > 0:
                cv2.putText(image, f"Tempo restante: {int(remaining_time)}s", (5, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(image, "Pressione 'Q' para sair", (5, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Exibe a imagem resultante
        cv2.imshow('MediaPipe Pose', image)

        # Verifica se alguma tecla foi pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera a captura e fecha as janelas abertas
cap.release()
cv2.destroyAllWindows()

print(f"\nAnotação da classe {current_class} finalizada!")
print(f"Total de imagens capturadas: {image_counter}")
print(f"Arquivos salvos em: {images_directory}")
print(f"CSV salvo em: {csv_file_path}")
