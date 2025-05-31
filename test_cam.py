import cv2

class Camera:
    def __init__(self, camera_index=6):
        self.camera_index = camera_index
        self.cap = None
        
    def iniciar(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Erro: não foi possível abrir a câmera no índice {self.camera_index}")
            return False
        return True
    
    def capturar_frame(self):
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def liberar(self):
        """
        Libera os recursos da câmera
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

def testar_camera_opencv(camera_index=6):
    #testar indices
    #v4l2-ctl --list-devices

    camera = Camera(camera_index)
    
    if not camera.iniciar():
        return
    
    print("Câmera iniciada com sucesso. Pressione 'q' para sair.")
    
    while True:
        ret, frame = camera.capturar_frame()
        if not ret:
            print("Falha ao capturar o frame.")
            break
        
        # Exibir o frame capturado
        cv2.imshow('RealSense (modo webcam via OpenCV)', frame)
        
        # Sair ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    camera.liberar()
    cv2.destroyAllWindows()
    print("Finalizado.")

if __name__ == "__main__":
    testar_camera_opencv()
