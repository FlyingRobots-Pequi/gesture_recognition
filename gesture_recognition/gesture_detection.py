import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import os
from collections import deque, Counter

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

class GestureDetector(Node):
    def __init__(self, model_path):
        super().__init__('gesture_detector')
        
        # Publishers e Subscribers
        self.publisher_ = self.create_publisher(String, 'gesture_detected', 10)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.process_image,
            10)
        
        # Inicializa o bridge para conversão de imagens
        self.bridge = CvBridge()
        
        # Carrega o modelo
        self.model = self.load_model(model_path)
        
        # Lista de classes
        self.lista_comandos = ["right", "left", "hold", "land",
                          "clockwise", "counter-clockwise", "up", "down",
                          "back", "return", "forward", "takeoff"]
        
        # Inicializa o mediapipe
        self.pose, self.mp_draw, self.pose_landmark_style = self.initialize_pose()
        
        self.get_logger().info('Gesture Detector Node iniciado')

        # Para robustez: janela deslizante dos últimos 5 frames
        self.recent_results = deque(maxlen=5)
        self.last_result = ""

    def load_model(self, model_path):
        model = Conv1DNet()
        model.load_state_dict(torch.load(model_path))
        self.get_logger().info(f'Modelo carregado de {model_path}')
        model.eval()
        return model

    def initialize_pose(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False,
                            model_complexity=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils
        drawing_styles = mp.solutions.drawing_styles
        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        return pose, mp_draw, pose_landmark_style

    def get_landmarks_data(self, results_pose):
        landmarks_data = {}
        landmark_names = [landmark.name for landmark in mp.solutions.pose.PoseLandmark]
        for i, landmark in enumerate(results_pose.pose_landmarks.landmark):
            if i not in list(range(0, 11)) + list(range(25, 33)):
                landmarks_data[f'{landmark_names[i]}.x'] = round(landmark.x, 4)
                landmarks_data[f'{landmark_names[i]}.y'] = round(landmark.y, 4)
                landmarks_data[f'{landmark_names[i]}.z'] = round(landmark.z, 4)
                landmarks_data[f'{landmark_names[i]}.visibility'] = round(landmark.visibility, 4)
        return landmarks_data

    def process_image(self, msg):
        try:
            # Converte a mensagem ROS para imagem OpenCV
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Processa a pose
            results_pose = self.pose.process(img_rgb)
            
            if results_pose.pose_landmarks:
                # Desenha os landmarks
                self.mp_draw.draw_landmarks(
                    img_rgb,
                    results_pose.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                # Obtém os pontos e faz a inferência
                landmarks_data = self.get_landmarks_data(results_pose)
                values_list = list(landmarks_data.values())
                values_array = np.array(values_list)
                new_data = torch.tensor(values_array, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
                
                with torch.no_grad():
                    prev = self.model(new_data)
                    probas = F.softmax(prev, dim=1)
                    pred_class = torch.argmax(probas, dim=1).item()
                    command_confidence = probas[0, pred_class].item()
                
                # Adiciona resultado à janela
                self.recent_results.append((pred_class, command_confidence))

                # Só publica/exibe se já houver 5 frames
                if len(self.recent_results) == 5:
                    predictions = [r[0] for r in self.recent_results]
                    final_pred_class = Counter(predictions).most_common(1)[0][0]
                    confidences_for_mode = [r[1] for r in self.recent_results if r[0] == final_pred_class]
                    avg_confidence = sum(confidences_for_mode) / len(confidences_for_mode)
                    threshold = 0.9
                    result_text = self.lista_comandos[final_pred_class]
                    if avg_confidence >= threshold:
                        # Publica o resultado
                        msg_pub = String()
                        msg_pub.data = result_text
                        self.publisher_.publish(msg_pub)
                        self.get_logger().info(f'Gesto detectado: {result_text} (confiança média: {avg_confidence:.2f})')
                        self.last_result = f"{result_text} - {round(avg_confidence, 2)}"
                    else:
                        self.last_result = f"Indeciso - {round(avg_confidence, 2)}"
                # Se ainda não houver 5 frames, mantém o último resultado
                result = self.last_result
                cv2.putText(img_rgb, result, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Converte de volta para BGR e publica a imagem processada
            img_display = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow('Gesture Detection', img_display)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Erro ao processar imagem: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    package_path = get_package_share_directory('gesture_recognition')
    model_path = os.path.join(package_path, 'conv1d.pth')
    gesture_detector = GestureDetector(model_path)

    try:
        rclpy.spin(gesture_detector)
    except KeyboardInterrupt:
        pass
    finally:
        gesture_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()