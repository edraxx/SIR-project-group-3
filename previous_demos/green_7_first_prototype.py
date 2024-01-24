# python C:\Users\hserdary\Desktop\VU\SIR\framework\sic_framework\services\dialogflow\dialogflow.py
import json
import queue
import numpy as np
from PIL import Image
from openai import OpenAI
from transformers import pipeline
from sic_framework.devices.nao import Nao
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.services.dialogflow.dialogflow import DialogflowConf, GetIntentRequest, Dialogflow, RecognitionResult, QueryResult

# # GLOBAL VARIABLES
ROBOT_IP = "192.168.0.121"
EMOTION_MODEL_PATH = "PriyamSheta/EmotionClassModel"
DIALOGFLOW_CREDENTIALS_FILE = "green_7_dialogflow_credentials.json"
OPENAI_CREDENTIALS_FILE = "hserdary_openai_credentials.txt"
OPENAI_MODEL = "gpt-4-1106-preview"
PRE_RECORDED_MOVEMENTS =  "[sit_down, stand_up, do_nothing]"
OPENAI_PROMPT = "You are a NAO robot. You will be given the text of speech and the emotion " + \
    "of the person in front of you inside a parentheses at the end of the text. You must give me an output " + \
    "consists of the short message you created, the sentiment score of the given input text between 0 and 1, " + \
    f"and decide a movement choice between {PRE_RECORDED_MOVEMENTS}. Always return as a json file consists of " + \
    "message:your_message and score:sentimen_score, movement:movement_choice. Always try to make the person happier " + \
    "to be supportive. Consider the sentiment_score and the given emotion of the person while creating answer."

def on_dialog(message):
    if message.response:
        if message.response.recognition_result.is_final:
            print("Transcript:", message.response.recognition_result.transcript)

def on_image(image_message: CompressedImageMessage):
    imgs_buffer.put(image_message.image)

with open(OPENAI_CREDENTIALS_FILE, "rb") as f:
    OPENAI_KEY = f.read().decode("utf-8").strip()

dialogflow_credentials = json.load(open(DIALOGFLOW_CREDENTIALS_FILE))
robot = Nao(ip=ROBOT_IP)
openai_client = OpenAI(api_key=OPENAI_KEY,)
conf = DialogflowConf(keyfile_json=dialogflow_credentials, sample_rate_hertz=16000, )
dialogflow = Dialogflow(ip='localhost', conf=conf)
dialogflow.register_callback(on_dialog)
dialogflow.connect(robot.mic)
print('# # deploying the model from HuggingFace.')
pipe = pipeline("image-classification", model=EMOTION_MODEL_PATH)
print('# # the model is deployed successfully.')
imgs_buffer = queue.Queue(maxsize=1) 
conf = NaoqiCameraConf(vflip=-1)
robot.top_camera.register_callback(on_image)

move_stand = NaoPostureRequest("Stand", 1.5)
move_sit = NaoPostureRequest("Sit", 1.5)
messages = [ {"role": "system", 
              "content": OPENAI_PROMPT} ]

robot.motion.request(move_stand)
robot.tts.request(NaoqiTextToSpeechRequest("Hi, I'm Nao. How can I help you?"))
print("# # # # Ready for Dialogflow + Emotion Classifier")
for i in range(50):    
    print("- - Conversation turn", i)
    reply = dialogflow.request(GetIntentRequest(np.random.randint(10000)))
    text = reply.response.query_result.query_text

    print('the emotion is being analysed...')
    img = imgs_buffer.get()
    image_model = pipe(Image.fromarray(img))
    print(f"EMOTION: {image_model[0]['label']}, CONF: {image_model[0]['score'].round(2)}") 

    if image_model[0]['score']>0.75:
        emotion = image_model[0]['label']
    else:
        emotion = 'Neutral'

    if text:
        print('Text:', text+f" ({emotion})")
        messages.append({'role':'user', 'content':text+f" ({emotion})"})
        chat = openai_client.chat.completions.create(model=OPENAI_MODEL, response_format={ "type": "json_object"}, messages=messages)
        reply = chat.choices[0].message.content
        if (len(reply)<20): continue # to check if whether it's empty
        reply_json = json.loads(reply)
        messages.append({"role": "assistant", "content": reply_json['message']}) 
        
        print('Nao Reply:', reply_json['message'])
        print('Sentiment Score:', reply_json['score'])
        robot.tts.request(NaoqiTextToSpeechRequest(reply_json['message']))

        if (reply_json['movement'] == 'sit_down'):
            print('Movement: sit_down')
            robot.motion.request(move_sit)
        elif (reply_json['movement'] == 'stand_up'):
            print('Movement: stand_up')
            robot.motion.request(move_stand)
        elif (reply_json['movement'] == 'do_nothing'):
            print('Movement: do_nothing')
        