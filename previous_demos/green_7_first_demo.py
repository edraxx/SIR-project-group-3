from sic_framework.devices import Nao, Pepper
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest # NaoRestRequest, NaoqiAutonomous
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiMotion, NaoPostureRequest, NaoqiAnimationRequest

nao = Pepper(ip='192.168.0.121')

# "Crouch", , "Sit", "SitRelax"
# "Stand", "StandInit", "StandZero"
# not working: "LyingBack" "LyingBelly"
move_stand = NaoPostureRequest("Stand", 1.5)
move_crouch = NaoPostureRequest("Crouch", 1.5)
move_sit = NaoPostureRequest("Sit", 1.5)
move_sit_relax = NaoPostureRequest("SitRelax", 1.5)
move_stand_zero = NaoPostureRequest("StandZero", 1.5)

nao.tts.request(NaoqiTextToSpeechRequest("Hey, I'm Nao and I will do some gestures for the group Green Seven today."))

nao.tts.request(NaoqiTextToSpeechRequest("I am going to stand"))
nao.motion.request(move_stand)

nao.tts.request(NaoqiTextToSpeechRequest("I am going to crouch"))
nao.motion.request(move_crouch)

nao.tts.request(NaoqiTextToSpeechRequest("I am going to sit and relax"))
nao.motion.request(move_sit_relax)

nao.tts.request(NaoqiTextToSpeechRequest("I am going to stand zero"))
nao.motion.request(move_stand_zero)

nao.tts.request(NaoqiTextToSpeechRequest("I am going to sit"))
nao.motion.request(move_sit)

nao.tts.request(NaoqiTextToSpeechRequest("Bye bye"))