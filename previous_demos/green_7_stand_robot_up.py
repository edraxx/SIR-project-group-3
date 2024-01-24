from sic_framework.devices.nao import Nao
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoBasicAwarenessRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest 

ROBOT_IP = "10.0.0.240"

robot = Nao(ip=ROBOT_IP)
move_stand = NaoPostureRequest("Stand", 1.5)

# robot.tts.request(NaoqiTextToSpeechRequest("Let's start."))
robot.autonomous.request(NaoBasicAwarenessRequest(False))
robot.motion.request(move_stand)