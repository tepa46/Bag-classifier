import os

from app.constants import datasetSavePath

datasetClasses = ["garbage", "paper", "plastic"]

datasetImageClassesPath = os.path.join(datasetSavePath, "Bag Classes/Bag Classes")

garbageBagsClassPath = os.path.join(datasetImageClassesPath, "Garbage Bag Images")
paperBagsClassPath = os.path.join(datasetImageClassesPath, "Paper Bag Images")
plasticBagsClassPath = os.path.join(datasetImageClassesPath, "Plastic Bag Images")

randomState = 42

datasetTestPart = 0.2

alpha = 0.05
