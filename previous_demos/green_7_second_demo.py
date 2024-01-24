import json
import time

import numpy as np
import pyaudio
# from sic_framework.devices.common_naoqi.nao_motion import NaoPostureRequest, NaoRestRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.nao import Nao

from sic_framework.services.dialogflow.dialogflow import DialogflowConf, \
    GetIntentRequest, RecognitionResult, QueryResult, Dialogflow


from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiMotion, NaoPostureRequest, NaoqiAnimationRequest, NaoqiMoveRequest

move_stand = NaoPostureRequest("Stand", 1.5)
move_crouch = NaoPostureRequest("Crouch", 1.5)
move_sit = NaoPostureRequest("Sit", 1.5)
move_sit_relax = NaoPostureRequest("SitRelax", 1.5)
move_stand_zero = NaoPostureRequest("StandZero", 1.5)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def on_dialog(message):
    if message.response:
        if message.response.recognition_result.is_final:
            print("Transcript:", message.response.recognition_result.transcript)


nao = Nao(ip='192.168.0.136')

keyfile_json = json.load(open("key_dialog.json"))
conf = DialogflowConf(keyfile_json=keyfile_json,
                      sample_rate_hertz=16000, )

dialogflow = Dialogflow(ip='localhost', conf=conf)
dialogflow.register_callback(on_dialog)
dialogflow.connect(nao.mic)

nao.tts.request(NaoqiTextToSpeechRequest("Hello!"))

print(" -- Ready -- ")
x = np.random.randint(10000)

for i in range(50):
    print(" ----- Conversation turn", i)
    reply = dialogflow.request(GetIntentRequest(x))

    print(reply.intent)

    if reply.fulfillment_message:
        text = reply.fulfillment_message
        print("Reply:", text)
        nao.tts.request(NaoqiTextToSpeechRequest(text))

    if reply.intent:
        print("Intent:", reply.intent)
        name = reply.response.query_result.intent.display_name
        if name == "stand_up_question":
            print("Standing up!")
            nao.motion.request(move_stand)
        elif name == "sit_relax":
            print("Sitting and relaxing!")
            nao.motion.request(move_sit_relax)
        elif name == "sit_down":
            print("Sitting down!")
            nao.motion.request(move_sit)
        elif name == "stand_zero":
            print("Standing Zero!")
            nao.motion.request(move_stand_zero)
        elif name == "ByeBye":
            break

time.sleep(100)