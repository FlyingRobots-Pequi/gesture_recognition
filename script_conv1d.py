import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Modelo Conv1DNet
class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (56 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (x.shape[2]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Carregar modelo
def load_model(model_path):
    model = Conv1DNet()
    model.load_state_dict(torch.load(model_path))
    print(f'Modelo carregado de {model_path}')
    model.eval()
    return model

# Inicializar o MediaPipe Pose
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

# Extrair landmarks do MediaPipe
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

# Função principal
def main():
    # Caminho do modelo
    model_path = r"C:\Users\Luisa Francielle\OneDrive\Área de Trabalho\Flying\conv1d.pth"
    model = load_model(model_path)

    lista_comandos = ["hold", "left", "right", "foward", "backward", 
                      "land", "takeoff", "up", "down", "clockwise", "counter_clockwise"]

    # Inicializar o MediaPipe
    pose, mp_draw, pose_landmark_style = initialize_pose()

    # Inicializar webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    with pose as pose:
        while True:
            ret, img_bgr = cap.read()
            if not ret:
                print("Erro ao capturar frame.")
                break

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(img_rgb)

            if results_pose.pose_landmarks:
                mp_draw.draw_landmarks(
                    img_rgb,
                    results_pose.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Obter landmarks
                landmarks_data = get_landmarks_data(results_pose)
                values_list = list(landmarks_data.values())
                values_array = np.array(values_list)
                new_data = torch.tensor(values_array, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

                # Inferência do modelo
                with torch.no_grad():
                    prev = model(new_data)
                    probas = F.softmax(prev, dim=1)
                    pred_class = torch.argmax(probas, dim=1).item()
                    command_confidence = probas[0, pred_class].item()

                # Mostrar comando na tela
                result_text = lista_comandos[pred_class]
                cv2.putText(img_bgr, f'{result_text} ({round(command_confidence, 2)})', 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostrar imagem
            cv2.imshow('MediaPipe Pose', img_bgr)

            # Sair com 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()