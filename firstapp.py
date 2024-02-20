from tkinter import *
from tkinter import filedialog
import cv2
import math
from PIL import Image, ImageTk
from ultralytics import YOLO
import cvzone
import pygame

pygame.mixer.init()

# --------------------------------------------
classnames = ['fire', 'smoke']
model = YOLO('best.pt')
# -----------------------------------------------

class ObjectDetectionApp:
    def __init__(self):
        self.make_app()
        self.vid = cv2.VideoCapture(0)
        self.width, self.height = 640, 480
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.create_widgets()

    def make_app(self):
        self.root = Tk()
        self.root.title('Fire Detection and Alarm Systems')
        self.root.geometry('1920x1040')
        self.root['bg'] = '#A52A2A'
        name = Label(self.root, text='FIRE DETECTION AND ALARM SYSTEMS', font=('Stencil', 20), bg='black',
                     fg='#800000')
        name.pack(pady=10)

    def create_widgets(self):
        self.label_widget = Label(self.root)
        self.label_widget.pack()

        self.btn_open_webcam = Button(self.root, text="WEBCAM", command=self.open_camera,
                                      font=('Calibri', 14), bg='gray', fg='black')

        self.btn_open_file = Button(self.root, text="Image and Video", command=self.open_file,
                                    font=('Calibri', 14), bg='gray', fg='black')

        self.btn_close_webcam = Button(self.root, text="StopCamera", command=self.exit_camera,
                                       font=('Calibri', 14), bg='gray', fg='black')

        self.btn_open_webcam.pack(pady=10)
        self.btn_open_file.pack(pady=10)
        self.btn_close_webcam.pack(pady=10)

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if "/" in file_path:
            print('Selected:', file_path)
            result = model(file_path, show=True)
            cv2.waitKey(0)

    def open_camera(self):
        if hasattr(self, "btn_open_camera"):
            self.btn_open_webcam["state"] = "disabled"
            self.btn_open_file["state"] = "disabled"

        _, frame = self.vid.read()
        result = model(frame, stream=True)

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])

                if Class < len(classnames) and confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    color = (0, 0, 255)

                    if Class == 1:
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                       scale=1.5, thickness=2)
                    pygame.mixer.Sound('sound.mp3').play()

        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)

        self.label_widget.photo_image = photo_image
        self.label_widget.configure(image=photo_image)

        self.root.after(10, self.open_camera)

    def exit_camera(self):
        if self.vid.isOpened():
            exit()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.run()
