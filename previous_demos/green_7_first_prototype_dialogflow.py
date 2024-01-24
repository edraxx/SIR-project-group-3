# python C:\Users\hserdary\Desktop\VU\SIR\framework\sic_framework\services\dialogflow\dialogflow.py
import json
import numpy as np
from openai import OpenAI
from sic_framework.devices.nao import Nao
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.services.dialogflow.dialogflow import DialogflowConf, GetIntentRequest, Dialogflow, RecognitionResult, QueryResult

ROBOT_IP = "192.168.0.121"
DIALOGFLOW_CREDENTIALS_FILE = "green_7_dialogflow_credentials.json"
OPENAI_CREDENTIALS_FILE = "hserdary_openai_credentials.txt"
OPENAI_MODEL = "gpt-4-1106-preview"
OPENAI_PROMPT = "You are a NAO robot. You must give me an output consists of the short message you created, the sentiment score of the given input, and decide a movement choice between (sit_down, stand_up, and do_nothing). Always return as a json file consists of message:your_message and score:sentimen_score, movement:movement_choice"

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

robot.tts.request(NaoqiTextToSpeechRequest("Ready"))
# robot.motion.request(move_stand)
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
        if (len(reply)<20):
            continue
        reply_json = json.loads(reply[8:-4])
        messages.append({"role": "assistant", "content": reply_json['message']}) 
        
        print('Reply:', reply_json['message'])
        print('Score:', reply_json['score'])
        # robot.tts.request(NaoqiTextToSpeechRequest(reply))

        if (reply_json['movement'] == 'sit_down'):
            print('Movement: sit_down')
            robot.motion.request(move_sit)
        elif (reply_json['movement'] == 'stand_up'):
            print('Movement: stand_up')
            robot.motion.request(move_stand)
        elif (reply_json['movement'] == 'do_nothing'):
            print('Movement: do_nothing')
        