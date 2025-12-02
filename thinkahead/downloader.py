from roboflow import Roboflow

rf = Roboflow(api_key="qdXCvI0ZQvIofMPsQ2eH")

project = rf.workspace("ducky-9ja6f").project("violations-hvg7q-7d1ba")
version = project.version(1)  # Check the version number on the website
dataset = version.download("yolov8", location="data/raw/roboflow_violations")