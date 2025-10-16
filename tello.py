import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from djitellopy import Tello
import time
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager

# Configuração de LOG
def setup_logging(level=logging.INFO, log_file="tello_run.log", max_bytes=5_000_000, backup_count=3):
    logger = logging.getLogger("tello")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

log = setup_logging()

# Modelo (Arquitetura)
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

# Utilidades
def load_model(model_path):
    model = Conv1DNet()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    log.info(f"Modelo carregado de '{model_path}'")
    return model

def initialize_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    log.info("MediaPipe Pose inicializado")
    return pose, mp_draw, pose_landmark_style

def get_landmarks_data(results_pose):
    landmarks_data = {}
    landmark_names = [landmark.name for landmark in mp.solutions.pose.PoseLandmark]
    for i, landmark in enumerate(results_pose.pose_landmarks.landmark):
        # mantém o mesmo filtro usado por você
        if i not in list(range(0, 11)) + list(range(25, 33)):
            landmarks_data[f'{landmark_names[i]}.x'] = round(landmark.x, 4)
            landmarks_data[f'{landmark_names[i]}.y'] = round(landmark.y, 4)
            landmarks_data[f'{landmark_names[i]}.z'] = round(landmark.z, 4)
            landmarks_data[f'{landmark_names[i]}.visibility'] = round(landmark.visibility, 4)
    return landmarks_data

@contextmanager
def tello_session():
    tello = Tello()
    try:
        tello.connect()
        log.info(f"Conectado ao Tello. Bateria: {tello.get_battery()}% | Temp: {tello.get_temperature()}°C")
        yield tello
    finally:
        try:
            log.info("Encerrando sessão: aterrissando e finalizando conexão.")
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.2)
            try:
                tello.land()
                time.sleep(1.0)
            except Exception as e:
                log.warning(f"Falha ao aterrissar no encerramento (pode já estar no solo): {e}")
        except Exception as e:
            log.error(f"Erro no teardown do Tello: {e}")
        finally:
            try:
                tello.end()
            except Exception as e:
                log.warning(f"Falha ao encerrar conexão Tello: {e}")

def log_rc(action, left_right=0, forward_back=0, up_down=0, yaw=0, extra=None, level=logging.INFO):
    msg = f"RC action='{action}' | lr={left_right} fb={forward_back} ud={up_down} yaw={yaw}"
    if extra:
        msg += f" | {extra}"
    log.log(level, msg)

# Função principal
def main():
    model_path = 'conv1d.pth'
    lista_comandos = [
        "right", "left", "hold", "land",
        "clockwise", "counter-clockwise", "up", "down",
        "back", "return", "forward", "takeoff"
    ]

    model = load_model(model_path)

    with tello_session() as tello:

        # Câmera
        try:
            tello.streamon()
            log.info("Stream da câmera ligado.")
        except Exception as e:
            log.error(f"Falha ao iniciar stream da câmera: {e}")
            return

        # Decolagem inicial
        try:
            tello.takeoff()
            #log.info("Decolagem realizada com sucesso.")
            #time.sleep(1.0)
            #tello.move_up(50) # sobe um pouco
            #time.sleep(4.0)
            #tello.move_right(200) #  direita
            #time.sleep(1.0)
            #tello.move_forward(50) #  frente
            #time.sleep(1.0)

        except Exception as e:
            log.error("Falha ao decolar. Possíveis causas: interferência, distância >2m, bateria baixa, Wi-Fi instável.")
            log.exception(e)
            return

        # MediaPipe
        pose, mp_draw, pose_landmark_style = initialize_pose()

        fps_t0 = time.perf_counter()
        fps_smooth = None

        try:
            with pose as pose_ctx:
                tello.send_rc_control(0, 0, 0, 0)
                log_rc("idle-before-loop")

                while True:

                    # Frame
                    frame = tello.get_frame_read().frame
                    if frame is None:
                        log.warning("Frame vazio recebido; mantendo posição.")
                        tello.send_rc_control(0, 0, 0, 0)
                        time.sleep(0.1)
                        continue

                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Pose
                    results_pose = pose_ctx.process(img_rgb)

                    if results_pose and results_pose.pose_landmarks:
                        # Desenho dos landmarks (apenas visual)
                        mp_draw.draw_landmarks(
                            img_rgb,
                            results_pose.pose_landmarks,
                            mp.solutions.pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                            connection_drawing_spec=mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                        )

                        # Extrai features para o modelo (sem cálculo de centro, sem follow)
                        landmarks_data = get_landmarks_data(results_pose)
                        values_array = np.array(list(landmarks_data.values()), dtype=np.float32)
                        new_data = torch.from_numpy(values_array).unsqueeze(0).unsqueeze(1)

                        # Inferência
                        with torch.no_grad():
                            logits = model(new_data)
                            probas = F.softmax(logits, dim=1)
                            pred_class = torch.argmax(probas, dim=1).item()
                            command_confidence = float(probas[0, pred_class])

                        # Threshold
                        threshold = 0.9
                        if command_confidence < threshold:
                            pred_class = 2  # "hold"
                            result_text = "hold"
                            shown_conf = 0.0
                            low_conf = True
                        else:
                            result_text = lista_comandos[pred_class]
                            shown_conf = round(command_confidence, 2)
                            low_conf = False

                        # Overlay e log
                        result_str = f"{result_text} - {shown_conf}"
                        cv2.putText(img_rgb, result_str, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        log.info(f"PRED='{result_text}' conf={shown_conf} low_conf={low_conf}")

                        # Execução do comando (sem ajuste automático de yaw)
                        try:
                            yaw = 0  # NÃO seguir/centralizar: yaw só muda nos comandos de giro
                            if pred_class == 0:  # right
                                tello.send_rc_control(30, 0, 0, yaw)
                                log_rc("right", left_right=30, yaw=yaw)
                                time.sleep(0.5)
                            elif pred_class == 1:  # left
                                tello.send_rc_control(-30, 0, 0, yaw)
                                log_rc("left", left_right=-30, yaw=yaw)
                                time.sleep(0.5)
                            elif pred_class == 2:  # hold
                                tello.send_rc_control(0, 0, 0, 0)
                                log_rc("hold")
                                time.sleep(0.5)
                            elif pred_class == 3:  # land
                                log.info("Comando: land")
                                try:
                                    tello.land()
                                    time.sleep(2.0)
                                    tello.takeoff()
                                    time.sleep(1.0)
                                    tello.send_rc_control(0, 0, 70, 0)
                                    log_rc("land->takeoff->rise", up_down=70)
                                    time.sleep(0.5)
                                except Exception as e:
                                    log.exception(f"Falha na sequência land/takeoff: {e}")
                            elif pred_class == 4:  # clockwise
                                tello.send_rc_control(0, 0, 0, 30)
                                log_rc("clockwise", yaw=30)
                                time.sleep(0.5)
                            elif pred_class == 5:  # counter-clockwise
                                tello.send_rc_control(0, 0, 0, -30)
                                log_rc("counter-clockwise", yaw=-30)
                                time.sleep(0.5)
                            elif pred_class == 6:  # up
                                tello.send_rc_control(0, 0, 20, 0)
                                log_rc("up", up_down=20)
                                time.sleep(1.0)
                            elif pred_class == 7:  # down
                                tello.send_rc_control(0, 0, -20, 0)
                                log_rc("down", up_down=-20)
                                time.sleep(0.5)
                            elif pred_class == 8:  # back
                                tello.send_rc_control(0, -30, 0, 0)
                                log_rc("back", forward_back=-30)
                                time.sleep(0.5)
                            elif pred_class == 9:  # return
                                log.info("Comando: return (land + end)")
                                try:
                                    tello.land()
                                    time.sleep(1.0)
                                except Exception as e:
                                    log.warning(f"Falha ao pousar no 'return': {e}")
                                break  # encerra o loop; teardown acontece no contextmanager
                            elif pred_class == 10:  # forward
                                tello.send_rc_control(0, 30, 0, 0)
                                log_rc("forward", forward_back=30)
                                time.sleep(0.5)
                            elif pred_class == 11:  # takeoff
                                log.info("Comando: takeoff")
                                try:
                                    tello.takeoff()
                                    time.sleep(0.5)
                                except Exception as e:
                                    log.exception(f"Falha em takeoff: {e}")
                            else:
                                tello.send_rc_control(0, 0, 0, 0)
                                log_rc("unknown-class-hold")
                                time.sleep(0.1)
                        except Exception as e:
                            log.exception(f"Erro ao enviar comando RC: {e}")
                            try:
                                tello.send_rc_control(0, 0, 0, 0)
                            except Exception:
                                pass
                            time.sleep(0.2)
                    else:
                        # Sem pessoa detectada → mantém posição
                        tello.send_rc_control(0, 0, 0, 0)
                        log_rc("no-person-hold")
                        cv2.putText(img_rgb, "No person detected - holding", (5, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        time.sleep(0.1)

                    # FPS (média suavizada)
                    now = time.perf_counter()
                    fps = 1.0 / max(now - fps_t0, 1e-6)
                    fps_t0 = now
                    fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)
                    cv2.putText(img_rgb, f"FPS: {fps_smooth:.1f}", (5, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                    # Overlay de instrução
                    cv2.putText(img_rgb, "Pressione 'Q' para sair.", (5, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

                    # Exibição
                    cv2.imshow('MediaPipe Pose', img_rgb)

                    # Saída pelo teclado
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        log.info("Tecla 'q' pressionada. Encerrando loop principal.")
                        try:
                            tello.land()
                            time.sleep(2)
                        except Exception as e:
                            log.warning(f"Falha ao pousar no encerramento por 'q': {e}")
                        break

        except KeyboardInterrupt:
            log.info("Interrupção por teclado (Ctrl+C) — iniciando encerramento seguro.")
        except Exception as e:
            log.exception(f"Erro no loop principal: {e}")
        finally:
            cv2.destroyAllWindows()
            log.info("Janela do OpenCV fechada.")

if __name__ == "__main__":
    main()