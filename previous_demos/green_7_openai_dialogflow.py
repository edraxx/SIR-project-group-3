import json
import numpy as np
from openai import OpenAI
from sic_framework.devices.nao import Nao
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.services.dialogflow.dialogflow import DialogflowConf, GetIntentRequest, Dialogflow # RecognitionResult, QueryResult
# python C:\Users\hserdary\Desktop\VU\SIR\framework\sic_framework\services\dialogflow\dialogflow.py

ROBOT_IP = "192.168.0.151"
DIALOGFLOW_CREDENTIALS_FILE = "green_7_dialogflow_credentials.json"
OPENAI_CREDENTIALS_FILE = "hserdary_openai_credentials.txt"
OPENAI_MODEL = "gpt-4-1106-preview"
OPENAI_PROMPT = "You are a NAO robot. You have to give very short answers to what you are asked for, at most 3 sentences."

def on_dialog(message):
    if message.response:
        if message.response.recognition_result.is_final:
            print("Transcript:", message.response.recognition_result.transcript)

with open(OPENAI_CREDENTIALS_FILE, "rb") as f:
    OPENAI_KEY = f.read().decode("utf-8").strip()
DIALOGFLOW_CREDENTIALS = json.load(open(DIALOGFLOW_CREDENTIALS_FILE))

robot = Nao(ip=ROBOT_IP)
openai_client = OpenAI(api_key=OPENAI_KEY,)
conf = DialogflowConf(keyfile_json=DIALOGFLOW_CREDENTIALS, sample_rate_hertz=16000, )
dialogflow = Dialogflow(ip='localhost', conf=conf)
dialogflow.register_callback(on_dialog)
dialogflow.connect(robot.mic)

move_stand = NaoPostureRequest("Stand", 1.5)
move_sit = NaoPostureRequest("Sit", 1.5)
messages = [ {"role": "system", 
              "content": OPENAI_PROMPT} ]

# robot.tts.request(NaoqiTextToSpeechRequest("Let's start!"))
robot.motion.request(move_stand)
print("# # # # Ready for Dialogflow")
for i in range(50):    
    print("- - Conversation turn", i)
    reply = dialogflow.request(GetIntentRequest(np.random.randint(10000)))
    text = reply.response.query_result.query_text
    if text:
        print('Text:', text)
        messages.append({'role':'user', 'content':text})
        chat = openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply}) 
        print('Reply:', reply)
        robot.tts.request(NaoqiTextToSpeechRequest(reply))