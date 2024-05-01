from roboflow import Roboflow

#Downloading the dataset
rf = Roboflow(api_key="NRlWBWvwgIMQKb99mDpF")
project = rf.workspace("corentin-f3le8").project("telesuiveur2")
version = project.version(1)
dataset = version.download("yolov8")