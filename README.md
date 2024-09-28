# Basic Multimodal Agentic Driver Assistant (MADA)



Next figure shows the functional blocks of MADA.

<img src="readme_files/esquema MADA.png" alt="MADA functional blocks" width="900" height="500" />

There are several sensors and processing modules:
- a camera that takes RGB and depth images. They are processed by the Object Detector in the computer, which detects objects (cars, traffic lights) and provides the object bounding box along with the mean distance from the camera
- a cell phone that gets the speed from the GPS module and sends it to the computer. It also recognizes driver requests and sends them as text to the computer. Moreover, the coordinates from the accelerometer and gyroscope are gathered and sent to the computer

All the events at the output of the processing modules are sent to the Driver Agent, which stores the information in the Memory and analyzes it in the Planner to assess if some action should be initiated. There are two types of actions:
- automatic actions: respond to one or more events that reflect some danger or warning. An example can be detecting that the distance to a car in front is lower than the safety distance.
- request motivated actions: respond to a speech request from the driver. An example could be checking if there is a safety distance with a bus in front. Those requests are sent to an LLM in the Planner which can select a function to be called.

## Code structure
The code is divided in two folders:
- **mada_android**: contains two project folders for two apps to be built in Android Studio and installed in an Android cell phone
- **mada_mac**: contains python files implementing the Driver Agent modules that run on a computer (a Mac M1, currently)

## MADA Android
There 

## MADA Mac
`object_detector.py`
`websocketServ.py`
`driver_agent.py`
`memory.py`
`planner.py`
`functions_schema.py`: automatically generates function schemas by parsing the function definitions in a given python file
`mada.yaml`: contains the configuration parameters
