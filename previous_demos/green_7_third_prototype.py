import json
import queue
from PIL import Image
from io import BytesIO
from openai import OpenAI
import speech_recognition as sr
from transformers import pipeline
from sic_framework.devices.nao import Nao
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoLEDRequest, NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import PlayRecording, NaoqiMotionRecording
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoBasicAwarenessRequest, NaoBlinkingRequest

# # GLOBAL VARIABLES
ROBOT_IP = "10.0.0.240"
IS_EMPATHY = True
IS_AWARENESS_MODE = True
LOOP_COUNTER = 50
EMOTION_MODEL_PATH = "dima806/facial_emotions_image_detection" # 5 different emotions: happy, sad, neutral, surprise, angry
DIALOGFLOW_CREDENTIALS_FILE = "green_7_dialogflow_credentials.json"
OPENAI_CREDENTIALS_FILE = "hserdary_openai_credentials.txt"
OPENAI_MODEL = "gpt-4-1106-preview"
WHISPER_MODEL = "whisper-1"
PRE_RECORDED_MOVEMENTS =  "[to_sit, to_stand, to_hug, to_clap, to_facepalm, to_celebrate, to_heart_react, to_open_up, being_shock, being_angry, being_scared, being_happy, doing_nothing]"
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
    "13)Emotional Discharge: denotes behavioral attempts to reduce the tension by expressing negative feelings. " +\
    "14)Following Up: Do not only give a reaction, but try to follow up on what the user says. Thus, have a conversation, not only an isolated reaction. "
NON_EMPATHY_PROMPT = "You are a NAO robot. You are a non-emotional entity, and you are uncapable of showing any empathy. " + \
    "You need to act like a robot, which cannot express its feelings. " + \
    "You must always return a json structured output consists of " + \
    "the short message you created to communicate with the person, " + \
    "the sentiment score of -1, " + \
    "the eye color of the robot as [1,0,0], " + \
    "the speed of your speech as 0.5, " +\
    f"and a movement choice between {PRE_RECORDED_MOVEMENTS}. " + \
    "The output structure must be exactly message:your_message, score:sentimen_score, rgb:rgb, speed:speed, movement:movement_choice."

def on_dialog(message):
    if message.response:
        if message.response.recognition_result.is_final:
            print("- DIALOGFLOW:", message.response.recognition_result.transcript)

def on_image(image_message: CompressedImageMessage):
    imgs_buffer.put(image_message.image)

def text_to_speech(client):
    print('+ recording...')
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    print('+ recording done!')
    wav_data = BytesIO(audio.get_wav_data())
    wav_data.name = "SpeechRecognition_audio.wav"
    text = client.audio.transcriptions.create(model=WHISPER_MODEL, file=wav_data, language='en').text
    return text

with open(OPENAI_CREDENTIALS_FILE, "rb") as f:
    OPENAI_KEY = f.read().decode("utf-8").strip()

# # configurations
dialogflow_credentials = json.load(open(DIALOGFLOW_CREDENTIALS_FILE))
conf = NaoqiCameraConf(vflip=-1, res_id=2, brightness=60)
robot = Nao(ip=ROBOT_IP, top_camera_conf=conf)
r = sr.Recognizer()
mic = sr.Microphone()
openai_client = OpenAI(api_key=OPENAI_KEY,)
print('# # deploying the model from HuggingFace.')
pipe = pipeline("image-classification", model=EMOTION_MODEL_PATH)
print('# # the model is deployed successfully.')
imgs_buffer = queue.Queue(maxsize=1) 
robot.top_camera.register_callback(on_image)
messages = [{"role": "system", 
             "content": OPENAI_PROMPT+EMPATHY_PROMPT if IS_EMPATHY else NON_EMPATHY_PROMPT}]
# # pre-recorded motions
to_sit = NaoPostureRequest("Sit", 1.5)
to_stand = NaoPostureRequest("Stand", 1.5)
to_hug = NaoqiMotionRecording.load('to_hug.motion')
to_clap = NaoqiMotionRecording.load('to_clap.motion')
to_facepalm = NaoqiMotionRecording.load('to_facepalm.motion')
to_celebrate = NaoqiMotionRecording.load('to_celebrate.motion')
to_heart_react = NaoqiMotionRecording.load('to_heart_react.motion')
to_open_up = NaoqiMotionRecording.load('to_open_up.motion')
being_shock = NaoqiMotionRecording.load('being_shock.motion')
being_angry = NaoqiMotionRecording.load('being_angry.motion')
being_scared = NaoqiMotionRecording.load('being_scared.motion')
being_happy   = NaoqiMotionRecording.load('being_happy.motion')
# # initializing the robot
robot.motion.request(to_stand)
robot.autonomous.request(NaoBlinkingRequest(IS_AWARENESS_MODE))
robot.autonomous.request(NaoBasicAwarenessRequest(IS_AWARENESS_MODE))
robot.leds.request(NaoLEDRequest("FaceLeds", True))
robot.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0, 1, 0))
robot.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0, 1, 0))
robot.tts.request(NaoqiTextToSpeechRequest("Hi, I'm Nao. How can I help you?" if IS_EMPATHY else "Hello there!" ))
# # conversation loop
print("# # # # ready for the conversation")
for i in range(LOOP_COUNTER):    
    first_img = imgs_buffer.get()
    first_emotion = pipe(Image.fromarray(first_img))
    print("# # ")
    print("- - conversation turn", i)
    text = text_to_speech(openai_client)
    second_img = imgs_buffer.get()
    second_emotion = pipe(Image.fromarray(second_img))
    if first_emotion[0]['score'] > second_emotion[0]['score']:
        print(f"+ emotion of the person: {first_emotion[0]['label']}, CONF: {first_emotion[0]['score']}") 
        emotion = first_emotion[0]['label'] if first_emotion[0]['score']>0.50 else 'Neutral'
    else:
        print(f"+ emotion of the person: {second_emotion[0]['label']}, CONF: {second_emotion[0]['score']}") 
        emotion = second_emotion[0]['label'] if second_emotion[0]['score']>0.50 else 'Neutral'
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
        # # motion acting
        if(reply_json['movement'] == 'to_sit'):
            print('+ nao movement: to_sit')
            robot.motion.request(to_sit)
        elif(reply_json['movement'] == 'to_stand'):
            print('+ nao movement: to_stand')
            robot.motion.request(to_stand)
        elif(reply_json['movement'] == 'to_hug'):
            print('+ nao movement: to_hug')
            robot.motion.request(to_stand)
            robot.motion_record.request(PlayRecording(to_hug))
        elif(reply_json['movement'] == 'to_clap'):
            print('+ nao movement: to_clap')
            robot.motion_record.request(PlayRecording(to_clap))
        elif(reply_json['movement'] == 'to_facepalm'):
            print('+ nao movement: to_facepalm')
            robot.motion_record.request(PlayRecording(to_facepalm))
        elif(reply_json['movement'] == 'to_celebrate'):
            print('+ nao movement: to_celebrate')
            robot.motion_record.request(PlayRecording(to_celebrate))   
        elif (reply_json['movement'] == 'to_heart_react'):
            print('+ nao movement: to_heart_react')
            robot.motion_record.request(PlayRecording(to_heart_react))
        elif (reply_json['movement'] == 'to_open_up'):
            print('+ nao movement: to_open_up')
            robot.motion_record.request(PlayRecording(to_open_up))
        elif (reply_json['movement'] == 'being_shock'):
            print('+ nao movement: being_shock')
            robot.motion_record.request(PlayRecording(being_shock))
        elif (reply_json['movement'] == 'being_angry'):
            print('+ nao movement: being_angry')
            robot.motion_record.request(PlayRecording(being_angry))
        elif (reply_json['movement'] == 'being_scared'):
            print('+ nao movement: being_scared')
            robot.motion_record.request(PlayRecording(being_scared))
        elif (reply_json['movement'] == 'being_happy'):
            print('+ nao movement: being_happy')
            robot.motion_record.request(PlayRecording(being_happy))
        else:
            print('+ nao movement: doing_nothing')
        # # eye coloring
        print('+ nao speed speech:',reply_json['speed'])
        print('+ nao eye rgb:',reply_json['rgb'])
        robot.leds.request(NaoFadeRGBRequest("LeftFaceLeds", float(reply_json['rgb'][0]), 
                                             float(reply_json['rgb'][1]), float(reply_json['rgb'][2]), 0))
        robot.leds.request(NaoFadeRGBRequest("RightFaceLeds", float(reply_json['rgb'][0]), 
                                             float(reply_json['rgb'][1]), float(reply_json['rgb'][2]), 0))