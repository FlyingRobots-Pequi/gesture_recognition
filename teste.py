from djitellopy import Tello
import time

def reboot_tello():
    tello = Tello()

    print("[INFO] Conectando ao Tello...")
    tello.connect()

    battery = tello.get_battery()
    print(f"[INFO] Bateria atual: {battery}%")

    tello.reboot()

if __name__ == "__main__":
    reboot_tello()