# voice-Command-Robot-Simulation
This repository is to create a simulation of voice to command to a robot in Ubuntu system
# Install requirement by
# pip install -r requirement.txt
# Hardware requirements:
#   PC(laptop) or embedded chip such as RK3588 chip.
#   System: Ubuntu(Linux) 
#   Microphone, either pc embedded microphone or extended USB microphone. 
#              Ubuntu(Linux) Command: arecord -l 
#                           can find available microphone device  
#              Example output: 
#              card 1: PCH [HDA Intel PCH], device 0: ALC233 Analog [ALC233 Analog]
#              子设备: 1/1
#              子设备 #0: subdevice #0
#             card 2: Audio [UGREEN CM564 USB Audio], device 0: USB Audio [USB Audio]
#              子设备: 1/1
#              子设备 #0: subdevice #0
#   
# Robot such as car-robot with IP address: Example: 192.168.4.1

# Usage: python3 voiceCommandRobot_v5.py --device 2 --robot-ip 192.168.4.1 --distance-factor 0.08 --turn-factor-left 1.8 --turn-factor-right 1.5 

# voiceCommandRobot_v4.py is the earlier version.
# Usage: python3 voiceCommandRobot_v4.py --device 2 --robot-ip 192.168.4.1 



