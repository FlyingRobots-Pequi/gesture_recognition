import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
import os

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

class GestureDetectorWebcam(Node):
    def __init__(self, model_path, camera_index=0):
        super().__init__('gesture_detector_webcam')
        
        # Publisher para gestos detectados
        self.publisher_ = self.create_publisher(String, 'gesture_detected', 10)
        
        # Configuração da webcam
        self.camera_index = camera_index
        self.cap = None
        
        # Carrega o modelo
        self.model = self.load_model(model_path)
        
        # Lista de classes (mesma ordem do treinamento)
        self.lista_comandos = ["right", "left", "hold", "land",
                          "clockwise", "counter-clockwise", "up", "down",
                          "back", "return", "forward", "takeoff"]
        
        # Inicializa o mediapipe
        self.pose, self.mp_draw, self.pose_landmark_style = self.initialize_pose()
        
        # Inicializa a webcam
        self.initialize_webcam()
        
        # Timer para processar frames da webcam
        self.timer = self.create_timer(0.03, self.process_webcam_frame)  # ~30 FPS
        
        # Controle de detecção para evitar spam
        self.last_detection_time = 0
        self.detection_cooldown = 1.0  # 1 segundo entre detecções
        
        self.get_logger().info('Gesture Detector Webcam Node iniciado')
        self.get_logger().info('Comandos disponíveis: takeoff, land, forward, back, left, right, up, down, hold, return')

    def initialize_webcam(self):
        """Inicializa a webcam"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Não foi possível abrir a câmera {self.camera_index}")
            
            # Configura resolução da webcam
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.get_logger().info(f'Webcam {self.camera_index} inicializada com sucesso')
            
        except Exception as e:
            self.get_logger().error(f'Erro ao inicializar webcam: {str(e)}')
            self.cap = None

    def load_model(self, model_path):
        """Carrega o modelo de detecção de gestos"""
        model = Conv1DNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.get_logger().info(f'Modelo carregado de {model_path}')
        model.eval()
        return model

    def initialize_pose(self):
        """Inicializa o MediaPipe Pose"""
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
        """Extrai dados dos landmarks da pose"""
        landmarks_data = {}
        landmark_names = [landmark.name for landmark in mp.solutions.pose.PoseLandmark]
        for i, landmark in enumerate(results_pose.pose_landmarks.landmark):
            if i not in list(range(0, 11)) + list(range(25, 33)):
                landmarks_data[f'{landmark_names[i]}.x'] = round(landmark.x, 4)
                landmarks_data[f'{landmark_names[i]}.y'] = round(landmark.y, 4)
                landmarks_data[f'{landmark_names[i]}.z'] = round(landmark.z, 4)
                landmarks_data[f'{landmark_names[i]}.visibility'] = round(landmark.visibility, 4)
        return landmarks_data

    def process_webcam_frame(self):
        """Processa um frame da webcam"""
        if self.cap is None or not self.cap.isOpened():
            return
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning('Não foi possível capturar frame da webcam')
                return
            
            # Converte BGR para RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processa a pose
            results_pose = self.pose.process(img_rgb)
            
            result_text = "Aguardando gesto..."
            command_confidence = 0.0
            
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
                
                # Verifica a certeza da previsão
                threshold = 0.9
                if command_confidence >= threshold:
                    result_text = self.lista_comandos[pred_class]
                    
                    # Controle de cooldown para evitar spam
                    current_time = time.time()
                    if current_time - self.last_detection_time > self.detection_cooldown:
                        # Publica o resultado
                        msg = String()
                        msg.data = result_text
                        self.publisher_.publish(msg)
                        self.get_logger().info(f'Gesto detectado: {result_text} (confiança: {command_confidence:.2f})')
                        self.last_detection_time = current_time
                else:
                    result_text = "Indeciso"
            
            # Adiciona texto informativo na imagem
            img_display = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Informações no topo
            cv2.putText(img_display, f"Gesto: {result_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Confiança: {command_confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instruções
            cv2.putText(img_display, "Pressione 'q' para sair", (10, img_display.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Mostra o frame
            cv2.imshow('Gesture Detection - Webcam', img_display)
            
            # Verifica se o usuário quer sair
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Encerrando detecção de gestos...')
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Erro ao processar frame da webcam: {str(e)}')

    def destroy_node(self):
        """Cleanup ao destruir o nó"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Localiza o modelo
        package_path = get_package_share_directory('gesture_recognition')
        model_path = os.path.join(package_path, 'conv1d.pth')
        
        # Verifica se o modelo existe
        if not os.path.exists(model_path):
            print(f"Modelo não encontrado em: {model_path}")
            # Tenta caminho alternativo
            model_path = os.path.join(os.path.dirname(__file__), 'conv1d.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), 'conv1d.pth')
        
        if not os.path.exists(model_path):
            print("Erro: Modelo não encontrado!")
            return
            
        # Cria o detector
        gesture_detector = GestureDetectorWebcam(model_path, camera_index=0)
        
        print("=== GESTURE DETECTION WEBCAM ===")
        print("Comandos disponíveis:")
        print("- takeoff: Iniciar voo")
        print("- land: Pousar")
        print("- forward/back: Movimento frente/trás")
        print("- left/right: Movimento esquerda/direita")
        print("- up/down: Movimento vertical")
        print("- hold: Manter posição")
        print("- return: Voltar para origem")
        print("- Pressione 'q' na janela para sair")
        print("===============================")
        
        rclpy.spin(gesture_detector)
        
    except KeyboardInterrupt:
        print("\nEncerrando por interrupção do usuário...")
    except Exception as e:
        print(f"Erro: {str(e)}")
    finally:
        if 'gesture_detector' in locals():
            gesture_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 