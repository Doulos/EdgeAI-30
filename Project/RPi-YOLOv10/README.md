Create a Docker image on Raspberry Pi that runs an object detection model using LiteRT.
The hardware setup consists of a Webcam connected to the Raspberry Pi board using a USB Hub.
Use the ideas on AI model output tensor, LiteRT runtime, MQTT and Docker are discussed in 
the course. 

**Step 1:**

Create a **src** folder containing the inference code, labels file and YOLO10 nano TFLite model file inside a directory
called **RPi-YOLOv10**. 

Use the [Ultralytics website](https://docs.ultralytics.com/modes/export) to export the Yolov10n model using 16 and 8 bit quantization options. 

Outside the src folder, create a requirements.txt file with the following Python packages 

- numpy
- ai-edge-litert
- opencv-python-headless
- paho-mqtt==1.6.1

Create a Docker file to use debian:bookworm-slim OS and Python 3.10-slim.  The slim versions are to make 
the image lightweight. 

**Step 2:**

Install Docker on Raspberry Pi using this command

$ curl -sSL https://get.docker.com | sh

Add your user to the `docker` group

$ sudo usermod -aG docker $USER

**Step 3:**

Create Docker image using build.

- Build image locally on the Raspberry Pi -3B and and expected to take about 3-4 minutes.
- Change to the RPi-YOLOv10 directory and issue the build command.

```bash
$sudo docker build -t yolov10-mqtt .
```

**Step 4:**

 Check if the image is successfully created

```bash
$docker images

REPOSITORY       TAG       IMAGE ID       CREATED          SIZE
yolov10-mqtt     latest    c73cb468959d   7 minutes ago    405MB

```

**Step 5:**
Run the Docker container created in step 3 in the background.  Camera is connected on the /dev/video0

```bash
$sudo docker run --privileged -v /dev/video0:/dev/video0 yolov10-mqtt &
```

**Step 6:**
Observe the output from the application. 

- The output is a MQTT payload consisting of object label, confidence score and bounding box coordinates.
- Sample outputs with camera pointing at a person. 

```bash
[3] 37984
pi@raspberrypi:~/RPi-YOLOv10$ INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Sending JSON: {"timestamp": 1766459874.8768485, "objects": {"class_id": 0, "label": "person", "confidence": 0.793, "bbox": [0.09138107299804688, 0.07930111885070801, 0.6551454067230225, 0.9981493949890137]}}
Inference published to MQTT topic
Logged to DB: person
MQTT client disconnected and loop stopped.
================================================================================
ID   | Timestamp            | Label      | Conf   | Coordinates (x1, y1, x2, y2)
--------------------------------------------------------------------------------
1    | 2025-12-23 03:17:54  | person     | 0.79   | (0.09138107299804688, 0.07930111885070801,
     |                      |            |        |  0.6551454067230225, 0.9981493949890137)
================================================================================



```

**Step 7:**

- Change the MQTT topic name in the application code (object-detection-yolov10.py) and subscribe to the topic on another computer to view the output from the object detection model.
- Try updating the code to give continuous inference.  Right now, it is setup to provide only one inference at start. 
- Use the SQLite database content to create a dashboard using Streamlit

