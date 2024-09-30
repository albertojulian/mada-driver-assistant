# Basic Multimodal Agentic Driver Assistant (MADA)

This MADA project's goal is having a minimum, but as functional as possible, **driver assistant** that works outdoor. Among the functionalities provided by MADA are:
- **safety distance checking**: if there is a vehicle in front, and the current distance is lower than the safety distance for the current speed, MADA warns with an audio message.
- **speed assistance**: when a speed limit traffic sign is detected, and the estimated speed is greater than the speed limit, MADA warns with an audio message.
- person (TODO)

### Why "multimodal" and "agentic"?
MADA is multimodal because it includes an LLM and is surrounded by speech input and output, RGB and depth images, speed data, accelerometer and gyroscope coordinates...
MADA can also be described as "agentic" because, although it has the components that correspond to an Agent (sensors, memory, planner, actions; hence the name of the Driver Agent module), it is not yet as functional to really be considered a complete Agent.

Regarding hardware, MADA is based in (and limited by) the following components:
- an Intel Realsense D435i RGB and Depth camera
- a Samsung cell phone
- an Apple M1 computer with (just) 8 GB of RAM

## Functional blocks
MADA is composed of several sensors and processing modules:
- the **camera** takes RGB and depth images. They are processed by the **Object Detector** in the computer, which detects objects (cars, traffic lights) and provides the object type, the bounding box and the position, along with the mean distance from the camera
- the **cell phone** gets the **speed** from the GPS, **recognizes driver speech requests**, gets the coordinates from the **accelerometer** and **gyroscope**, and sends all this data to the computer. On the other hand, the cell phone **provides wi-fi** to the computer.

All the data at the output of the processing modules are sent to the **Driver Agent**, which converts them into events to be stored in the Memory and analyzed in the Planner to assess if some action should be initiated.
The only outputs are speech audio warnings or suggestions. The approach is non-invasive: there is no intention to take control of the car, just to assist the driver with speech messages.

Next figure shows the functional blocks of MADA.

<img src="readme_files/esquema MADA.png" alt="MADA functional blocks" width="900" height="500" />

## Object Detector
The goal of an Object Detection model is to analyze an image and identify which object types, out of a given list, are there in the image (or frame), along with the bounding box of each object.

The first version of MADA is based on the base YOLO v8 Object Detector model; the selected size of the models is medium: nano does not detect well enough, and bigger models run too slow in the platform.
The base model was trained on the COCO 2017 dataset, which provides several types useful for MADA: person, car, bus, bicycle, truck, traffic light. In future versions of MADA the Object Detector will be customized to detect additional object types, mainly traffic signs.

### Object Tracking
After detection, tracking is performed to keep the objects uniquely identified in successive frames. Tracked objects and their associated space events are memorized in order to enable certain actions.

### Distance
The depth image of the Intel Realsense D435i camera has a point-cloud format, which is filtered with only the points inside the bounding box limits of the object instance in the RGB image. The depth of those points is averaged, giving an average distance from the camera to the object as a result.
It is worth mentioning that depth cameras have a confidence range of distances; in the case of the Intel Realsense D435i is 0.1 – 3 meters, although still has quiet accuracy up to 6-7 meters; anyway, it would be better to have a camera with a range up to 20 meters or so.

## Speed estimation

## Speech recognition
Speech recognition was initially performed by whisper.cpp in stream mode. However, it required 1 GB of RAM and I wanted to 
implement it in the same Kotlin app as the speed estimation. It uses the built-in capabilities of the Android cell phone.


## 

## Driver Agent
Next figure shows the Driver Agent structure.

(TODO) AÑADIR EJEMPLO DE USO DE DRIVER AGENT
<img src="readme_files/driver_agent.png" alt="Driver Agent structure" width="900" height="500" />

 There are two types of actions:
- **automatic actions**: respond to one or more events that correspond to some kind of risk or danger. An example can be calling a function to check if the distance to a car in front is lower than the safety distance.
- **request motivated actions**: respond to a speech request from the driver. An example could be checking if there is a safety distance with a bus in front. Those requests are sent to an LLM in the Planner which can select a function to be called.

## Memory
It enables the persistence of objects and events:
- Object: all the types that can be detected by the object detector: car, bus, person, traffic light, traffic sign. Defined by an object type and a track id (which remains in successive frames to uniquely identify the object instance)
- SpaceEvent: associated to an object instance. Defined by a bounding box, a position (left, front, right, depending on the center of the bounding box)
- SpeedEvent: (TODO)
- ActionEvent: mainly used to avoid repeating the same action over the same object too soon 

## Planner
(TODO) LLM Gemma2

## Code structure
The code is divided in two folders:
- **mada_android**: contains two project folders for two Android apps
- **mada_mac**: contains python files implementing the Driver Agent modules that run on a computer (a Mac M1, currently)

### MADA Android
Contains two project folders for two Kotlin apps to be built in Android Studio and installed in an Android cell phone: 
- **SpeedVoice**: gets the speed from the GPS and transforms the driver speech requests into text; then sends that data as webSocket messages to the webSockets server in the computer.
- **AccelGyro**: gets the coordinates from the **accelerometer** and **gyroscope**, and sends them as webSockets messages to the webSockets server in the computer

### MADA Mac
Contains the following python files:
- `object_detector.py`: implements object detection and tracking. In "live" mode it captures the RGB and Depth images from the camera; otherwise, it takes them from videos
- `websockets_server.py`: implements a webSockets server that receives websocket messages from the processing modules (apps in the cell phone and object detector in the computer) and converts them into events to be stored and processed
- `driver_agent.py`: implements the DriverAgent class, which contains the Memory and Planner classes; implements the Planner class, which manages the LLM that supports the **driver request motivated actions**; it also performs the evaluation of both automatic and request motivated actions
- `memory.py`: contains all class definitions to support the persistence of objects and events
- `functions.py`: includes the definition of the functions supporting the actions
- `planner.py`: 
- `functions_schema.py`: automatically generates function schemas by parsing the function definitions in `functions.py`. The schemas are used by the LLM in the Driver Agent's Planner to decide if one of the existing functions must be called. It also contains a FunctionParser class used by the Driver Agent's Planner to parse the JSON output of the LLM in function calling mode and ensure the function exists and is correctly called
- `text_to_speech.py`: manages the text-to-speech functionality, currently very simple: 
  - a call to Google's gtts service, which takes a text and delivers an audio file of the spoken text
  - a call to macOS `afplay` command, which takes an audio file and plays it
- `record_rgb_and_depth_videos.py`: records RGB and depth videos to support changes in the object detector indoor without having to connect the camera.

It also contains other files which the previous python files use:
- `yolov8m.pt`: base model used by the object detector
- `mada.yaml`: contains the configuration parameters that are used by the python functions and classes

## Execution instructions
- In the Cell Phone: turn on GPS and Shared Connection
- In the Computer:
  - connect the camera to the computer
  - link to the cell phone wi-fi
  - open a terminal, enter the virtual environment with the MADA packages and run the websockets server (which starts the Driver Agent): `python websocket_server.py`
  - open another terminal, enter the virtual environment with the MADA packages and run the Object Detector: `sudo python object_detector.py`
- In the Cell Phone: start the SpeedVoice and AccelGyro apps

## Future work
- Customize the base YOLO model to detect traffic signs
- Fine-tune the LLM model to enable more specific and complex requests
- Add left and right back cameras to track blind spots, in order to enable detection of vehicles in those positions when trying to move to left or right lane