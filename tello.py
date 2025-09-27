# Importações
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from djitellopy import Tello
import time

# Arquitetura Final
class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (56 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, 12) 
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (x.shape[2]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Carrega o modelo com os pesos treinados
def load_model(model_path):
    model = Conv1DNet()
    model.load_state_dict(torch.load(model_path))
    print(f'Modelo carregado de {model_path}')
    model.eval()
    return model

# Inicializa a captura de poses com o mediapipe
def initialize_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    return pose, mp_draw, pose_landmark_style

# Obtém os pontos de interesse
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
    elif section_width <= center_x <= 3 * section_width:
        print("Pessoa está no centro, permanecendo no lugar")
        yaw = 0
    return yaw

# Função principal
def main():

    # Carrega o modelo
    model_path = 'conv1d.pth'
    model = load_model(model_path)
    
    # Lista de classes atualizada
    lista_comandos = ["right", "left", "hold", "land",
                      "clockwise", "counter-clockwise", "up", "down",
                      "back", "return", "forward", "takeoff"]
    
    # Inicializa os movimentos do drone
    tello = Tello()
    tello.connect()
    tello.streamon() # Inicializa a câmera do drone
    # time.sleep(3.0)

    tello.takeoff() # Inicializa o voo do drone
    time.sleep(1.0)
    tello.send_rc_control(0, 0, 55, 0) # Define uma altura para o drone inicializar a captura de movimentos
    time.sleep(1.0)
    
    # Inicializa o mediapipe
    pose, mp_draw, pose_landmark_style = initialize_pose()
    
    with pose as pose:
        tello.send_rc_control(0, 0, 0, 0) # Encerra qualquer movimento do drone para entrar no loop principal

        # Loop principal
        while True:

            # Captura a imagem e obtém os pontos de interesse
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
                
                # Calcula a posição central dos ombros para girar o drone caso necessário
                landmarks_data = get_landmarks_data(results_pose)
                left_shoulder_x = landmarks_data['LEFT_SHOULDER.x'] * frame_width
                right_shoulder_x = landmarks_data['RIGHT_SHOULDER.x'] * frame_width
                center_x = (left_shoulder_x + right_shoulder_x) / 2
    
                yaw = turn_drone(center_x, frame_width)

                # Obtém os pontos atuais
                values_list = list(landmarks_data.values())
                values_array = np.array(values_list)
                new_data = torch.tensor(values_array, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    
                # Faz inferência com o modelo treinado
                with torch.no_grad():
                    prev = model(new_data)
                    probas = F.softmax(prev, dim=1)
                    pred_class = torch.argmax(probas, dim=1).item()
                    command_confidence = probas[0, pred_class].item()
    
                # Verifica a certeza da previsão
                result_text = lista_comandos[pred_class]
                threshold = 0.9
                default = 2
                if command_confidence < threshold:
                    pred_class = 2  # "hold" como padrão
                    result_text = "hold"
                    result = f"{result_text} - {default}"
                else:
                    result = f"{result_text} - {round(command_confidence, 2)}"

                # Manipulação da imagem de log
                cv2.putText(img_rgb, result, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
                # Executa um comando no drone baseado na previsão
                if pred_class == 0:  # right
                    print('right')
                    tello.send_rc_control(30, 0, 0, yaw)
                    time.sleep(1.0)
                elif pred_class == 1:  # left
                    print('left')
                    tello.send_rc_control(-30, 0, 0, yaw)
                    time.sleep(1.0)
                elif pred_class == 2:  # hold
                    print('hold')
                    tello.send_rc_control(0, 0, 0, yaw)
                    time.sleep(1.0)
                elif pred_class == 3:  # land
                    print('land')
                    tello.takeoff()
                    time.sleep(1.0)
                elif pred_class == 4:  # clockwise
                    print('clockwise')
                    tello.send_rc_control(0, 0, 0, 30)
                    time.sleep(1.0)
                elif pred_class == 5:  # counter-clockwise
                    print('counter-clockwise')
                    tello.send_rc_control(0, 0, 0, -30)
                    time.sleep(1.0)
                elif pred_class == 6:  # up
                    print('up')
                    tello.send_rc_control(0, 0, 20, yaw)
                    time.sleep(1.0)
                elif pred_class == 7:  # down
                    print('down')
                    tello.send_rc_control(0, 0, -20, yaw)
                    time.sleep(1.0)
                elif pred_class == 8:  # back
                    print('back')
                    tello.send_rc_control(0, -30, 0, yaw)
                    time.sleep(1.0)
                elif pred_class == 9:  # return
                    print('return')
                    # Implementação para retornar ao ponto inicial
                    tello.send_rc_control(0, 0, 0, yaw)
                    time.sleep(1.0)
                elif pred_class == 10:  # forward
                    print('forward')
                    tello.send_rc_control(0, 30, 0, yaw)
                    time.sleep(1.0)
                elif pred_class == 11:  # takeoff
                    print('takeoff')
                    tello.takeoff()
                    time.sleep(1.0)
    
            # Manipulação da imagem de log
            cv2.putText(img_rgb, "Pressione 'Q' para sair.", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('MediaPipe Pose', img_rgb)
    
            # Encerra a aplicação pressionando a tecla "q"
            if cv2.waitKey(5) & 0xFF == ord('q'):
                tello.land()
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()