import time
from sic_framework.devices.pepper import Pepper
from sic_framework.devices.common_naoqi.naoqi_stiffness import Stiffness
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiIdlePostureRequest
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import StartRecording, StopRecording, PlayRecording, NaoqiMotionRecorderConf

ROBOT_IP = "192.168.0.151"

MOTION_RECORD_TIME = 15
BODY_PARTS = ['LArm', 'RArm'] # ['Body', 'Legs', 'Arms', 'LArm', 'RArm', 'Head']
MOTION_FILE_NAME = "gesture_file_name_here.motion"

conf = NaoqiMotionRecorderConf(use_sensors=True, 
                               use_interpolation=True, 
                               samples_per_second=60)
robot = Pepper(ROBOT_IP, motion_record_conf=conf)

robot.motion.request(NaoqiIdlePostureRequest("Body", False))
robot.stiffness.request(Stiffness(0.0, BODY_PARTS))
time.sleep(2)
print("Starting to record in one second!")
time.sleep(1)
robot.motion_record.request(StartRecording(BODY_PARTS))
print("Start moving the robot!")
time.sleep(MOTION_RECORD_TIME)
recording = robot.motion_record.request(StopRecording())
recording.save(MOTION_FILE_NAME)
print("Done.")
time.sleep(2)

print("Replaying the recorded action.")
robot.stiffness.request(Stiffness(.95, BODY_PARTS))
robot.motion_record.request(PlayRecording(recording))
robot.stiffness.request(Stiffness(.0, BODY_PARTS))