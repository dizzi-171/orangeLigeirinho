from threading import Thread
import cv2

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise ValueError("Não foi possível abrir a câmera.")
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()
