import time
from sic_framework.devices import Pepper
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest # NaoRestRequest, NaoqiAutonomous
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiMotion, NaoPostureRequest, NaoqiAnimationRequest
from sic_framework.services.dialogflow.dialogflow import Dialogflow, DialogflowConf, GetIntentRequest
from sic_framework.devices.common_naoqi.naoqi_stiffness import Stiffness
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import PlayRecording, NaoqiMotionRecorderConf, NaoqiMotionRecording

robot = Pepper(ip='10.0.0.240')
# move_sit = NaoPostureRequest("Sit", 1.5)
move_stand = NaoPostureRequest("Stand", 1.5)

being_angry = NaoqiMotionRecording.load('being_angry.motion')
being_happy   = NaoqiMotionRecording.load('being_happy.motion')
being_scared = NaoqiMotionRecording.load('being_scared.motion')
being_shock = NaoqiMotionRecording.load('being_shock.motion')
to_celebrate = NaoqiMotionRecording.load('to_celebrate.motion')
to_clap = NaoqiMotionRecording.load('to_clap.motion')
to_facepalm = NaoqiMotionRecording.load('to_facepalm.motion')
to_heart_react = NaoqiMotionRecording.load('to_heart_react.motion')
to_hug = NaoqiMotionRecording.load('to_hug.motion')
to_open_up = NaoqiMotionRecording.load('to_open_up.motion')

robot.motion.request(move_stand)
print('READY')
robot.motion_record.request(PlayRecording(being_angry))
print(1)
robot.motion_record.request(PlayRecording(being_happy))
print(2)
robot.motion_record.request(PlayRecording(being_scared))
print(3)
robot.motion_record.request(PlayRecording(being_shock))
print(4)
robot.motion_record.request(PlayRecording(to_celebrate))
print(5)
robot.motion_record.request(PlayRecording(to_clap))
print(6)
robot.motion_record.request(PlayRecording(to_facepalm))
print(7)
robot.motion_record.request(PlayRecording(to_heart_react))
print(8)
robot.motion_record.request(PlayRecording(to_hug))
print(9)
robot.motion_record.request(PlayRecording(to_open_up))
print(10)