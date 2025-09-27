# test_tello_min.py
from djitellopy import Tello

t = Tello()
t.connect()                  # NÃO envie 'command' manualmente em outro lugar
print("Conectado. Bateria:", t.get_battery())
t.end()
