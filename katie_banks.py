import json
import queue
import wave
from PIL import Image
from io import BytesIO
from openai import OpenAI
import speech_recognition as sr
from transformers import pipeline
from sic_framework.devices.nao import Nao
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.core.message_python2 import AudioMessage, AudioRequest
from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoLEDRequest, NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import PlayRecording, NaoqiMotionRecording
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoBasicAwarenessRequest, NaoBlinkingRequest

# # GLOBAL VARIABLES
ROBOT_IP = "10.0.0.210"
IS_EMPATHY = True # true for empathy mode
AMBIENT = False # true for loudy environment
IS_AWARENESS_MODE = True # true for empathy mode
LOOP_COUNTER = 50 # max conversation number
EMOTION_MODEL_PATH = "dima806/facial_emotions_image_detection" # 5 different emotions: happy, sad, neutral, surprise, angry
OPENAI_CREDENTIALS_FILE = "OPENAI_CREDENTIALS_HERE.txt"
OPENAI_MODEL = "gpt-4-1106-preview"
WHISPER_MODEL = "whisper-1"
PRE_RECORDED_MOVEMENTS =  "[to_sit, to_stand, to_hug, to_clap, to_facepalm, to_celebrate, to_heart_react, being_shock, being_angry, being_scared, being_happy, doing_nothing]"
OPENAI_PROMPT = "You will be given the text of the speech and the captured emotion of the person (inside the paranthesis at the end of the text). " +\
    "While creating the output, always only consider the sentiment score of the given speech."
EMPATHY_PROMPT = "This experiment requires you to continue the conversation with a user. The user is confiding in you on a personal matter. Listen with empathy. Avoid coming off as judgemental or apathetic. Talk with the user aout Katie Banks' story or maybe any way to help Katie Banks." +\
    "You can get benefit from the speed of your speech" +\
    "You must always return a json structured output consists of " +\
    "the short message you created to communicate with the person, " +\
    "the sentiment score of the given text in the continuous scale between 0 and 1, " +\
    "the speed of your speech in the continuous scale between 0 and 1 (0.5 is regular speed), " +\
    "The output structure must be exactly message:your_message, score:sentimen_score, speed:speed."
NON_EMPATHY_PROMPT = "This experiment requires you to continue the conversation with a user. The user is confiding in you on a personal matter. Do not empathize. The conversation revolves around the sad Katie Banks' story. Talk with the user about how much they are willing to donate in terms of hours of volunteering." +\
    "You must always return a json structured output consists of " +\
    "the short message you created to communicate with the person, " +\
    "the sentiment score of -1, " +\
    "the speed of your speech as 0.5, " +\
    "The output structure must be exactly message:your_message, score:sentimen_score, speed:speed."


#OPENAI_PROMPT = "This experiment requires you to continue the conversation with a user. The user is confiding in you on a personal matter. Listen with empathy. Avoid coming off as judgemental or apathetic. You must always return a json structured output consists of the message you created to communicate with the person. The sentiment score of the given text in the continuous scale between 0 and 1. The speed of your speech in the continuous scale between 0 and 1 (0.5 is regular speed). The output structure must be exactly message:your_message, score:sentimen_score, speed:speed. While creating the output, always only consider the sentiment score of the given speech."

wavefile = wave.open('./sounds/first.wav', 'rb')
samplerate = wavefile.getframerate()
sound = wavefile.readframes(wavefile.getnframes())
message_first = AudioMessage(sample_rate=samplerate, waveform=sound)

wavefile = wave.open('./sounds/second.wav', 'rb')
samplerate = wavefile.getframerate()
sound = wavefile.readframes(wavefile.getnframes())
message_second = AudioMessage(sample_rate=samplerate, waveform=sound)

def on_image(image_message: CompressedImageMessage):
    imgs_buffer.put(image_message.image)
    
def text_to_speech(client,ambient):
    global message_first, message_second
    print('+ recording...')
    robot.speaker.send_message(message_first)
    if ambient==True:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
    else:
        with mic as source:
            audio = r.listen(source)
    robot.speaker.send_message(message_second)
    print('+ recording done!')
    wav_data = BytesIO(audio.get_wav_data())
    wav_data.name = "SpeechRecognition_audio.wav"
    text = client.audio.transcriptions.create(model="whisper-1", file=wav_data,language='en').text
    return text

with open(OPENAI_CREDENTIALS_FILE, "rb") as f:
    OPENAI_KEY = f.read().decode("utf-8").strip()

# # configurations
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
to_hug = NaoqiMotionRecording.load('./motions/to_hug.motion')
to_clap = NaoqiMotionRecording.load('./motions/to_clap.motion')
to_facepalm = NaoqiMotionRecording.load('./motions/to_facepalm.motion')
to_celebrate = NaoqiMotionRecording.load('./motions/to_celebrate.motion')
to_heart_react = NaoqiMotionRecording.load('./motions/to_heart_react.motion')
being_shock = NaoqiMotionRecording.load('./motions/being_shock.motion')
being_angry = NaoqiMotionRecording.load('./motions/being_angry.motion')
being_scared = NaoqiMotionRecording.load('./motions/being_scared.motion')
being_happy   = NaoqiMotionRecording.load('./motions/being_happy.motion')
# # initializing the robot
robot.motion.request(to_stand)
robot.autonomous.request(NaoBlinkingRequest(IS_AWARENESS_MODE))
robot.autonomous.request(NaoBasicAwarenessRequest(IS_AWARENESS_MODE))
robot.leds.request(NaoLEDRequest("FaceLeds", True))
robot.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0, 1, 0))
robot.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0, 1, 0))
robot.tts.request(NaoqiTextToSpeechRequest("Katie is trying desperately to keep her family together and to finish school, but many problems confront her. She does not have enough money, she needs sitters to stay with her brother and sister  and transportation but does not have a car. Katie is trying to raise money through private contributions. What do you think of this story?"))
#Last week a tragic accident struck the Banks family of Lawrence, Kansas. Mr. and Mrs. George Banks and their 16-year-old daughter Jeanette were killed in a head-on collision 30 miles west of Wichita. The Banks family has lived in Lawrence for only 6 months. They were returning to their former home in Burton, Kansas, to visit friends. Mr. and Mrs. Banks left three surviving children—Katie, a senior at the University of Kansas; Alice, age 11; and Mark, age 8. Katie has been given temporary guardianship of her younger brother and sister. Unfortunately, Mr. Banks did not carry life insurance, and the children were left with very little money. Katie is trying desperately to keep her family together and to finish school. She hopes to graduate this summer, but many problems confront her. She does not have enough money for groceries or rent. She needs sitters to stay with her brother and sister while she attends her classes. And she needs transportation to the grocery store, laundry, and to school since she does not have a car. Katie is trying to raise money through private contributions.
# # conversation loop
print("# # # # ready for the conversation")
for i in range(LOOP_COUNTER):
    first_img = imgs_buffer.get()
    first_emotion = pipe(Image.fromarray(first_img))
    print("# # ")
    print("- - conversation turn", i)
    text = text_to_speech(openai_client,AMBIENT)
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
#        if text.lower().contains("bye-bye"):
#            i = 50
        messages.append({'role':'user', 'content':text+f" ({emotion})"})
        chat = openai_client.chat.completions.create(model=OPENAI_MODEL, response_format={ "type": "json_object"}, messages=messages)
        reply = chat.choices[0].message.content
        if (len(reply)<23): continue # to check whether it's empty
        reply_json = json.loads(reply)
        messages.append({"role": "assistant", "content": reply_json['message']})
        print('+ sentiment score of the person:', reply_json['score'])
        print('- NAO.......:', reply_json['message'])
        robot.tts.request(NaoqiTextToSpeechRequest(reply_json['message'], animated=IS_AWARENESS_MODE))
        print('+ nao speed speech:',reply_json['speed'])

