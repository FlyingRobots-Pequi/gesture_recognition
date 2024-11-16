# Importações
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from djitellopy import Tello
import time

# Definição da classe Conv1DNet
class Conv1DNet(nn.Module):
    def __init__(self, input_size):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        conv_output_size = self.calculate_conv_output_size(input_size)
        self.fc1 = nn.Linear(32 * conv_output_size, 64)
        self.fc2 = nn.Linear(64, 11)

    def calculate_conv_output_size(self, input_size):
        size = input_size
        size = (size + 2 * 1 - 3) // 1 + 1  # Após conv1
        size = size // 2  # Após pool1
        size = (size + 2 * 1 - 3) // 1 + 1  # Após conv2
        size = size // 2  # Após pool2
        return size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * x.shape[2])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Função para carregar o modelo
def load_model(model_path, input_size):
    model = Conv1DNet(input_size)
    model.load_state_dict(torch.load(model_path))
    print(f'Modelo carregado de {model_path}')
    model.eval()
    return model

# Inicializa a captura de poses com o MediaPipe
def initialize_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,         # Vídeo em tempo real
        model_complexity=2,              # Modelo mais robusto
        min_detection_confidence=0.7,    # Maior confiança para detecção inicial
        min_tracking_confidence=0.7,     # Maior confiança para rastreamento
        smooth_landmarks=True,           # Suavizar os landmarks
        enable_segmentation=True,        # Ativar segmentação do corpo
        smooth_segmentation=True         # Suavizar a segmentação
    )
    mp_draw = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    return pose, mp_draw, pose_landmark_style

# Função para obter os dados dos landmarks
def get_landmarks_data(results_pose):
    landmarks_data = {}
    landmark_names = [landmark.name for landmark in mp.solutions.pose.PoseLandmark]
    for i, landmark in enumerate(results_pose.pose_landmarks.landmark):
        if i not in list(range(0, 11)) + list(range(25, 33)):
            landmarks_data[f'{landmark_names[i]}.x'] = round(landmark.x, 4)
            landmarks_data[f'{landmark_names[i]}.y'] = round(landmark.y, 4)
            landmarks_data[f'{landmark_names[i]}.z'] = round(landmark.z, 4)
            landmarks_data[f'{landmark_names[i]}.visibility'] = round(landmark.visibility, 4)
    return landmarks_data

# Função para rotacionar o drone
def turn_drone(center_x, frame_width):
    section_width = frame_width / 4
    yaw = 0
    if center_x < section_width:
        print("Virando no sentido anti-horário")
        yaw = -30
    elif center_x > 3 * section_width:
        print("Virando no sentido horário")
        yaw = 30
    else:
        print("Pessoa está no centro, permanecendo no lugar")
        yaw = 0
    return yaw

# Função principal
def main():
    # Carrega o modelo
    model_path = r"C:\Users\Luisa Francielle\OneDrive\Área de Trabalho\Flying\conv1d.pth"
    input_size = 56  # Defina o input_size apropriado
    model = load_model(model_path, input_size)
    
    lista_comandos = ["hold", "left", "right", "forward", "backward",
                      "land", "takeoff", "up", "down", "clockwise", "counter_clockwise"]
    
    # Inicializa o drone e a câmera
    tello = Tello()
    tello.connect()
    tello.streamon()  # Inicializa a câmera do drone

    # Variável para rastrear o estado do drone
    is_flying = False

    try:
        # Decolagem do drone
        tello.takeoff()
        is_flying = True
        time.sleep(2)

        # Faz o drone subir 100 cm (1 metro)
        tello.move_up(100)
        time.sleep(2)  # Aguarda 2 segundos para garantir que o drone subiu

        tello.send_rc_control(0, 0, 0, 0)  # Garante que o drone está estável
        time.sleep(1)
        
        # Inicializa o MediaPipe
        pose, mp_draw, pose_landmark_style = initialize_pose()
        
        # Variáveis para controle de comandos
        current_command = 0  # Começa com 'hold'
        last_pred_class = None
        command_counter = 0
        command_threshold = 5  # Número de frames que o novo gesto deve ser consistente

        # Loop principal
        while True:
            # Captura a imagem da câmera do drone
            img_bgr = tello.get_frame_read().frame
            frame_height, frame_width = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(img_rgb)

            if results_pose.pose_landmarks:
                mp_draw.draw_landmarks(
                    img_rgb,
                    results_pose.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                # Processamento dos landmarks
                landmarks_data = get_landmarks_data(results_pose)
                left_shoulder_x = landmarks_data['LEFT_SHOULDER.x'] * frame_width
                right_shoulder_x = landmarks_data['RIGHT_SHOULDER.x'] * frame_width
                center_x = (left_shoulder_x + right_shoulder_x) / 2

                # Cálculo do yaw para manter a pessoa centralizada
                yaw = turn_drone(center_x, frame_width)

                # Preparação dos dados para o modelo
                values_list = list(landmarks_data.values())
                values_array = np.array(values_list)
                new_data = torch.tensor(values_array, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

                # Inferência do modelo
                with torch.no_grad():
                    prev = model(new_data)
                    probas = F.softmax(prev, dim=1)
                    pred_class = torch.argmax(probas, dim=1).item()
                    command_confidence = probas[0, pred_class].item()

                # Verificação da confiança da previsão
                result_text = lista_comandos[pred_class]
                threshold = 0.9
                default = 2
                if command_confidence < threshold:
                    pred_class = 0
                    result_text = "hold"
                    result = f"{result_text} - {default}"
                else:
                    result = f"{result_text} - {round(command_confidence, 2)}"

                # Exibição do resultado na imagem
                cv2.putText(img_rgb, result, (5, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)

                # Lógica para mudança de comando apenas após confirmação
                if pred_class == current_command:
                    # Se o comando previsto é o mesmo que está em execução, zera o contador
                    command_counter = 0
                else:
                    # Se o comando previsto é diferente do atual
                    if pred_class == last_pred_class:
                        command_counter += 1
                    else:
                        command_counter = 1  # Reinicia o contador
                        last_pred_class = pred_class

                    if command_counter >= command_threshold:
                        # Atualiza o comando atual
                        current_command = pred_class
                        command_counter = 0  # Reinicia o contador

                # Executa o comando atual no drone
                if current_command == 0:  # Hold
                    print('Hold')
                    tello.send_rc_control(0, 0, 0, yaw)
                elif current_command == 1:  # Left
                    print('Esquerda')
                    tello.send_rc_control(-20, 0, 0, yaw)
                elif current_command == 2:  # Right
                    print('Direita')
                    tello.send_rc_control(20, 0, 0, yaw)
                elif current_command == 3:  # Forward
                    print('Frente')
                    tello.send_rc_control(0, 20, 0, yaw)
                elif current_command == 4:  # Backward
                    print('Trás')
                    tello.send_rc_control(0, -20, 0, yaw)
                elif current_command == 5:  # Land
                    if is_flying:
                        print('Pousar')
                        tello.land()
                        is_flying = False
                        break  # Sai do loop após pousar
                elif current_command == 6:  # Takeoff
                    if not is_flying:
                        print('Decolar')
                        tello.takeoff()
                        is_flying = True
                        time.sleep(2)
                        tello.move_up(100)  # Sobe novamente após decolar
                        time.sleep(2)
                elif current_command == 7:  # Up
                    print('Subir')
                    tello.send_rc_control(0, 0, 20, yaw)
                elif current_command == 8:  # Down
                    print('Descer')
                    tello.send_rc_control(0, 0, -20, yaw)
                elif current_command == 9:  # Clockwise
                    print('Girar horário')
                    tello.send_rc_control(0, 0, 0, 30)
                elif current_command == 10:  # Counter-clockwise
                    print('Girar anti-horário')
                    tello.send_rc_control(0, 0, 0, -30)

                # Pequena pausa para estabilidade
                time.sleep(0.1)

            else:
                # Se nenhum gesto for detectado, manter o drone parado
                tello.send_rc_control(0, 0, 0, 0)

            # Exibição da imagem
            cv2.putText(img_rgb, "Pressione 'Q' para sair.", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('MediaPipe Pose', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            # Encerra a aplicação pressionando a tecla "q"
            if cv2.waitKey(5) & 0xFF == ord('q'):
                if is_flying:
                    tello.land()
                break

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        if is_flying:
            tello.land()
    finally:
        # Libera os recursos
        cv2.destroyAllWindows()
        tello.streamoff()  # Encerra a transmissão de vídeo
        tello.end()        # Desconecta do drone

if __name__ == "__main__":
    main()