import queue
import time
from PIL import Image
from transformers import pipeline
from sic_framework.devices.desktop import Desktop
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices.common_desktop.desktop_camera import DesktopCameraConf

EMOTION_MODEL_PATH = "PriyamSheta/EmotionClassModel"
# # we need to find the best fitted model and parameter combinations for the project
# https://huggingface.co/nateraw/vit-age-classifier
# https://huggingface.co/PriyamSheta/EmotionClassModel 
# https://huggingface.co/CynthiaCR/emotions_classifier
# https://huggingface.co/jayanta/microsoft-resnet-50-cartoon-emotion-detection

print('# # deploying the model from HuggingFace.')
pipe = pipeline("image-classification", model=EMOTION_MODEL_PATH)
print('# # the model is deployed successfully.')

def on_image(image_message: CompressedImageMessage):
    imgs_buffer.put(image_message.image)

imgs_buffer = queue.Queue(maxsize=1) 
conf = DesktopCameraConf(fx=0.5, fy=0.5, flip=-1)
desktop = Desktop(camera_conf=conf)
desktop.camera.register_callback(on_image)

print('# # READY')
while True:
    img = imgs_buffer.get()
    image_model = pipe(Image.fromarray(img))
    print(f"EMOTION: {image_model[0]['label']}, CONF: {image_model[0]['confidence']}") 

    # cv2.imshow('', img)
    # cv2.waitKey(1)
    # time.sleep(1)
    print('# # # # #')