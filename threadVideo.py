from threading import Thread
import cv2
import time

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.fps = 0

        if not self.cap.isOpened():
            raise ValueError("Não foi possível abrir a câmera.")
                             
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        deuPau = False
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
            else: 
                if not deuPau:
                    print("deu pau")
                    deuPau = True
                    
            time.sleep(0.025)
            self.fps = 1 / (time.time() - start_time)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()
