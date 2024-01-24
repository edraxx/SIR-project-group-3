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
from sic_framework.devices.common_naoqi.naoqi_leds import NaoLEDRequest, NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import PlayRecording, NaoqiMotionRecording
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoBasicAwarenessRequest, NaoBlinkingRequest
from sic_framework.services.dialogflow.dialogflow import DialogflowConf, GetIntentRequest, Dialogflow, RecognitionResult, QueryResult

# # GLOBAL VARIABLES
ROBOT_IP = "192.168.0.136"
IS_AWARENESS_MODE = True
EMOTION_MODEL_PATH = "dima806/facial_emotions_image_detection" # 5 different emotions: happy, sad, neutral, fear, angry
DIALOGFLOW_CREDENTIALS_FILE = "green_7_dialogflow_credentials.json"
OPENAI_CREDENTIALS_FILE = "hserdary_openai_credentials.txt"
OPENAI_MODEL = "gpt-4-1106-preview"
PRE_RECORDED_MOVEMENTS =  "[being_angry, being_happy, being_scared, being_shocked, acting_heart_reaction, acting_clapping, acting_facepalm, acting_celebrate, acting_hugging, acting_sitting_down, acting_standing_up, doing_nothing]"
OPENAI_PROMPT = "You are a NAO robot who can feel the emotions and be capable of empathizing. Your main task is to emphatize with the " + \
    "person in front of you and show him/her your emotions by talking with him/her and acting some pre-recorded movements provided to you. " + \
    "You will be given the text of the speech and the captured emotion of the person (inside the paranthesis at the end of the text). " + \
    "Make the sentiment analysis of the given speech, match your feelings with the person and always try to show your emotions and to empathy. " + \
    "While creating the output, always consider the sentiment score of the given speech and the emotion of the person. " + \
    "You can get benefit from your eye color, the speed of your speech, and movement to impact the person. " + \
    "You must always return a json structured output consists of " + \
    "the short message you created to communicate with the person, " + \
    "the sentiment score of the given text in the continuous scale between 0 and 1, " + \
    "the eye color of the robot to show your emotions using RGB 0-1 scale as [r,g,b], " + \
    "the speed of your speech in the continuous scale between 0 and 1 (0.5 is regular speed), " +\
    f"and a movement choice between {PRE_RECORDED_MOVEMENTS} to act like a human while talking. " + \
    "The output structure must be exactly message:your_message, score:sentimen_score, rgb:rgb, speed:speed, movement:movement_choice."
EMPATHY_PROMPT = "Below, you can see the key terms of empathy. Take them into account while generating the output. " +\
    "1)Encouraging Comments: for example, 'Don't not be sad, I believe you can still recover your disadvantage.'. " +\
    "2)Mimical: 'I'm happy that you are happy!'. " +\
    "3)Alleviational: which try to reduce the user's distress through reactive empathy, for instance: 'You did your best, don't regret.'. " +\
    "4)Motivational: which are a form of reactive empathy that motivate users to pass the current negative emotion, such as: 'Let it go, look for next round'. " +\
    "5)Distractional: Distractional, which can distract users from negative emotions, for example: 'Do you know what day is today?'. " +\
    "6)Logical Analysis: a cognitive strategy designed to understand and prepare oneself mentally to cope with the stressor and its consequences. " +\
    "7)Positive Reappraisal: aims to reconstruct the problem in a positive form while accepting the reality of the situation. " +\
    "8)Seeking Guidance and Support: covers behavioral attempts to seek information, guidance, or support to deal with the stressor. " +\
    "9)Problem Solving: denotes behavioral attempts to deal directly with the problem and solve it. " +\
    "10)Cognitive Avoidance: refers to cognitive attempts to avoid thinking about the problem or stressor. " +\
    "11)Acceptance or Resignation: refers to cognitive attempts to respond to the problem by accepting it and resigning oneself to it, because nothing can be done about it. " +\
    "12)Seeking Alternative Rewards: denotes behavioral attempts to alleviate the effect of the stress caused by the problem by seeking new forms of satisfaction. " +\
    "13)Emotional Discharge: denotes behavioral attempts to reduce the tension by expressing negative feelings."

def on_dialog(message):
    if message.response:
        if message.response.recognition_result.is_final:
            print("- DIALOGFLOW:", message.response.recognition_result.transcript)

def on_image(image_message: CompressedImageMessage):
    imgs_buffer.put(image_message.image)

with open(OPENAI_CREDENTIALS_FILE, "rb") as f:
    OPENAI_KEY = f.read().decode("utf-8").strip()

dialogflow_credentials = json.load(open(DIALOGFLOW_CREDENTIALS_FILE))
conf = NaoqiCameraConf(vflip=-1)
robot = Nao(ip=ROBOT_IP, top_camera_conf=conf)
openai_client = OpenAI(api_key=OPENAI_KEY,)
conf = DialogflowConf(keyfile_json=dialogflow_credentials, sample_rate_hertz=16000, )
dialogflow = Dialogflow(ip='localhost', conf=conf)
dialogflow.register_callback(on_dialog)
dialogflow.connect(robot.mic)
print('# # deploying the model from HuggingFace.')
pipe = pipeline("image-classification", model=EMOTION_MODEL_PATH)
print('# # the model is deployed successfully.')
imgs_buffer = queue.Queue(maxsize=1) 
robot.top_camera.register_callback(on_image)

move_sit = NaoPostureRequest("Sit", 1.5)
move_stand = NaoPostureRequest("Stand", 1.5)
move_angry = NaoqiMotionRecording.load('angry.motion')
move_happy   = NaoqiMotionRecording.load('happiness.motion')
move_scared = NaoqiMotionRecording.load('scared.motion')
move_shock = NaoqiMotionRecording.load('shocked.motion')
move_clap = NaoqiMotionRecording.load('clap.motion')
move_hugging = NaoqiMotionRecording.load('hug.motion')
move_heart_react = NaoqiMotionRecording.load('heart_reach.motion')
move_facepalm = NaoqiMotionRecording.load('facepalm.motion')
move_celebrate = NaoqiMotionRecording.load('celebrate.motion')
# move_greetings = NaoqiMotionRecording.load('greet.motion')
# move_understand = NaoqiMotionRecording.load('open_up.motion')
messages = [{"role": "system", 
             "content": OPENAI_PROMPT+EMPATHY_PROMPT}]

robot.motion.request(move_stand)
robot.autonomous.request(NaoBlinkingRequest(IS_AWARENESS_MODE))
robot.autonomous.request(NaoBasicAwarenessRequest(IS_AWARENESS_MODE))
robot.leds.request(NaoLEDRequest("FaceLeds", True))
robot.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0, 1, 0))
robot.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0, 1, 0))
robot.tts.request(NaoqiTextToSpeechRequest("Hi, I'm Nao. How can I help you?"))
print("# # # # Ready to emphatize")
for i in range(50):    
    print("# # ")
    print("- - Conversation turn", i)
    reply = dialogflow.request(GetIntentRequest(np.random.randint(10000)))
    text = reply.response.query_result.query_text

    img = imgs_buffer.get()
    image_model = pipe(Image.fromarray(img))
    print(f"+ emotion of the person: {image_model[0]['label']}, CONF: {image_model[0]['score']}") 
    emotion = image_model[0]['label'] if image_model[0]['score']>0.75 else 'Neutral'

    if text:
        print('- PERSON:....', text+f" ({emotion})")
        messages.append({'role':'user', 'content':text+f" ({emotion})"})
        chat = openai_client.chat.completions.create(model=OPENAI_MODEL, response_format={ "type": "json_object"}, messages=messages)
        reply = chat.choices[0].message.content
        if (len(reply)<23): continue # to check whether it's empty
        reply_json = json.loads(reply)
        messages.append({"role": "assistant", "content": reply_json['message']}) 
        print('+ sentiment score of the person:', reply_json['score'])
        print('- NAO.......:', reply_json['message'])
        robot.tts.request(NaoqiTextToSpeechRequest(reply_json['message'], animated=IS_AWARENESS_MODE))

        if (reply_json['movement'] == 'acting_sitting_down'):
            print('+ nao movement: acting_sitting_down')
            robot.motion.request(move_sit)
        elif (reply_json['movement'] == 'acting_standing_up'):
            print('+ nao movement: acting_standing_up')
            robot.motion.request(move_stand)
        elif (reply_json['movement'] == 'acting_hugging'):
            print('+ nao movement: acting_hugging')
            robot.motion.request(move_stand)
            robot.motion_record.request(PlayRecording(move_hugging))
        elif (reply_json['movement'] == 'being_angry'):
            print('+ nao movement: being_angry')
            # robot.motion.request(move_stand)
            robot.motion_record.request(PlayRecording(move_angry))
        elif (reply_json['movement'] == 'being_happy'):
             print('+ nao movement: being_happy')
             robot.motion.request(move_stand)
             robot.motion_record.request(PlayRecording(move_happy))
        elif (reply_json['movement'] == 'being_shocked'):
             print('+ nao movement: being_shocked')
             robot.motion.request(move_stand)
             robot.motion_record.request(PlayRecording(move_shock))   
        elif (reply_json['movement'] == 'being_scared'):
            print('+ nao movement: being_scared')
            robot.motion.request(move_stand)
            robot.motion_record.request(PlayRecording(move_scared))
        elif (reply_json['movement'] == 'acting_facepalm'):
            print('+ nao movement: acting_facepalm')
            # robot.motion.request(move_stand)
            robot.motion_record.request(PlayRecording(move_facepalm))
        elif (reply_json['movement'] == 'acting_celebrate'):
            print('+ nao movement: acting_celebrate')
            robot.motion.request(move_stand)
            robot.motion_record.request(PlayRecording(move_celebrate))
        elif (reply_json['movement'] == 'acting_heart_reaction'):
            print('+ nao movement: acting_heart_reaction')
            robot.motion.request(move_stand)
            robot.motion_record.request(PlayRecording(move_heart_react))
        elif (reply_json['movement'] == 'acting_clapping'):
            print('+ nao movement: acting_clapping')
            robot.motion.request(move_stand)
            robot.motion_record.request(PlayRecording(move_clap))
        elif (reply_json['movement'] == 'doing_nothing'):
            print('+ nao movement: doing_nothing')

        print('+ nao speed speech:',reply_json['speed'])
        print('+ nao eye rgb:',reply_json['rgb'])
        robot.leds.request(NaoFadeRGBRequest("LeftFaceLeds", float(reply_json['rgb'][0]), 
                                             float(reply_json['rgb'][1]), float(reply_json['rgb'][2]), 0))
        robot.leds.request(NaoFadeRGBRequest("RightFaceLeds", float(reply_json['rgb'][0]), 
                                             float(reply_json['rgb'][1]), float(reply_json['rgb'][2]), 0))