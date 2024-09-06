# Object Detection using OpenCV and MobileNet-SSD

[Watch the video](https://www.linkedin.com/posts/abhinav-varshney-bb9bb7204_connections-computervision-python-activity-7237905832913297408-orR-?utm_source=share&utm_medium=member_desktop)

## Table of Contents

## Introduction
This project demonstrates real-time **object detection** using the **MobileNet-SSD** deep learning model implemented with **OpenCV's DNN module** in Python. It detects objects in live video feed using a pre-trained model from the **COCO dataset** and highlights the objects with bounding boxes.

The key goal of this project is to enable real-time inference, making it suitable for applications like surveillance systems, autonomous robots, and real-time analytics.

## Tech Stack
- **Python**: The core programming language used to integrate OpenCV and machine learning models.
- **OpenCV**: The library for real-time computer vision, handling video processing and object detection.
- **MobileNet-SSD**: A lightweight deep learning architecture for efficient object detection.
- **COCO Dataset**: A large-scale dataset for object detection with 80 common object categories.

## Installation
To get started, clone this repository and install the required dependencies:

1. Clone the repository:
    ```bash
    git clone https://github.com/imabhnv/objectDetection.git
    ```

2. Navigate to the project directory:
    ```bash
    cd objectDetection
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

    **Note**: Ensure you have OpenCV installed. If not, you can install it via:
    ```bash
    pip install opencv-python
    pip install opencv-python-headless  # If GUI support is not needed
    ```

## Usage
1. Download the necessary files:
   - [COCO class names]
   - [MobileNet-SSD config file]
   - [MobileNet-SSD pre-trained weights]

   Place them in the appropriate folder, as specified in the code.

2. Run the object detection script:
    ```bash
    python object_detection.py
    ```

3. To quit the video stream, press the `q` key.

## Model and Dataset
- **Model**: The project utilizes the **SSD MobileNet v3 Large** model pre-trained on the **COCO** dataset.
- **Dataset**: COCO (Common Objects in Context) contains 80 object categories including people, animals, and everyday objects.

### Configuration Paths:
Ensure that the paths to the model and class names are correctly set in your code:
```python
classFile = 'path/to/coco.names'
configPath = 'path/to/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'path/to/frozen_inference_graph.pb'
