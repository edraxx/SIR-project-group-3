# python C:\Users\hserdary\Desktop\VU\SIR\framework\sic_framework\services\face_recognition_dnn\face_recognition.py
import cv2
import queue
from PIL import Image
from transformers import pipeline
from PIL import Image
from sic_framework.devices.desktop import Desktop
from sic_framework.core.utils_cv2 import draw_bbox_on_image
from sic_framework.devices.common_desktop.desktop_camera import DesktopCameraConf
from sic_framework.services.face_recognition_dnn.face_recognition import DNNFaceRecognition
from sic_framework.core.message_python2 import BoundingBoxesMessage, CompressedImageMessage

# https://huggingface.co/nateraw/vit-age-classifier
pipe = pipeline("image-classification", model="nateraw/vit-age-classifier")

imgs_buffer = queue.Queue(maxsize=1)
faces_buffer = queue.Queue(maxsize=1)

def on_image(image_message: CompressedImageMessage):
    imgs_buffer.put(image_message.image)

def on_faces(message: BoundingBoxesMessage):
    faces_buffer.put(message.bboxes)

conf = DesktopCameraConf(fx=0.75, fy=0.75, flip=-1)
desktop = Desktop(camera_conf=conf)
face_rec = DNNFaceRecognition()
face_rec.connect(desktop.camera)
desktop.camera.register_callback(on_image)
face_rec.register_callback(on_faces)

print('# # READY')
while True:
    img = imgs_buffer.get()
    faces = faces_buffer.get()
    ages = pipe(Image.fromarray(img))

    print("AGE INTERVAL: ", ages[0]['label'])

    for face in faces:
        draw_bbox_on_image(face, img)

    cv2.imshow('', img)
    cv2.waitKey(1)
    print('# # # # #')