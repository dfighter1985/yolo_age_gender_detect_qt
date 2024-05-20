This application demonstrates age and gender detection from a face image using Yolov8 Python API in a PyQt desktop application

## Installation

```
python -mvenv qt
cd qt
Scripts\activate.bat
pip install pyqt6
pip install ultralytics
```

Copy the Yolo age and gender detection model to this directory with name best.pt.

## Usage

Start the application with

```python yolo_age_gender_detect_qt.py```

Then click browse and select a cropped face image.

NOTE: Model is not included in the repository. See the following blog post for dataset to use for training:
https://dfighter1985.wordpress.com/2024/05/20/converting-the-utkface-computer-vision-dataset-to-the-yolo-format/


## Screenshot

![Screenshot](https://dfighter1985.wordpress.com/wp-content/uploads/2024/05/20240520_000038478.jpg)

