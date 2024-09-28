# Basic Multimodal Agentic Driver Assistant (MADA)

The goal is having a driver assistant as complete as possible with only a few elements:

The only output is speech audio warnings or suggestions.

Next figure shows the functional blocks of MADA.

<img src="readme_files/esquema MADA.png" alt="MADA functional blocks" width="900" height="500" />

There are several sensors and processing modules:
- a **camera** that takes RGB and depth images. They are processed by the **Object Detector** in the computer, which detects objects (cars, traffic lights) and provides the object class, the bounding box and the position, along with the mean distance from the camera
- a **cell phone** that gets the **speed** from the GPS and sends it to the computer. It also **recognizes driver speech requests** and sends them as text to the computer. Moreover, the coordinates from the **accelerometer** and **gyroscope** are gathered and sent to the computer

<img src="readme_files/driver_agent.png" alt="Driver Agent structure" width="500" height="350" />

All the data at the output of the processing modules are sent to the **Driver Agent**, which converts them into events 
to be stored in the Memory and analyzed in the Planner to assess if some action should be initiated. There are two types of actions:
- **automatic actions**: respond to one or more events that reflect some danger or warning. An example can be detecting that the distance to a car in front is lower than the safety distance.
- **request motivated actions**: respond to a speech request from the driver. An example could be checking if there is a safety distance with a bus in front. Those requests are sent to an LLM in the Planner which can select a function to be called.

## Code structure
The code is divided in two folders:
- **mada_android**: contains two project folders for two apps to be built in Android Studio and installed in an Android cell phone
- **mada_mac**: contains python files implementing the Driver Agent modules that run on a computer (a Mac M1, currently)

## MADA Android
There are two apps: 
- **SpeedVoice**: gets the speed from the GPS and sends it as a webSocket message to the webSockets server in the computer.
- **AccelGyro**: 

## MADA Mac
- `object_detector.py`: realsense, point-cloud, YOLO, Ultralytics tracker
- `websocketServ.py`: implements a webSockets server that receives websocket messages from the processing modules (apps in the cell phone and object detector in the computer) and converts them into events to be processed 
- `driver_agent.py`
- `memory.py`: contains all class definitions to support the persistence of objects and events:
  - Memory
  - Object
  - SpaceEvent
  - SpeedEvent
  - ActionEvent: mainly used to avoid repeating the same action over the same object too soon 
- `planner.py`
- `functions_schema.py`: automatically generates function schemas by parsing the function definitions in a given python file. The schemas are used by the LLM in the Driver Agent's Planner to decide if a given function must be called. It also contains a FunctionParser class used by the Driver Agent's Planner to parse the JSON output of the LLM in function calling mode and ensure the function exists and is correctly called
- `mada.yaml`: contains the configuration parameters
