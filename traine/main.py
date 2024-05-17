import os
from ultralytics import YOLO
from IPython.display import display, Image
from IPython import display
from roboflow import Roboflow
import multiprocessing


if __name__ == '__main__':
    display.clear_output()

    rf = Roboflow(api_key="nw7oVdMmrAoiTwr7odWA")
    project = rf.workspace("martin-kozr").project("vehicles-cns4f")
    version = project.version(3)
    dataset = version.download("yolov8")

    multiprocessing.freeze_support()
    model = YOLO("yolov8n.pt")
    model.train(data='./vehicles-3/data.yaml', epochs=2)