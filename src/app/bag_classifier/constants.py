import os

datasetKagglePath = "vencerlanz09/plastic-paper-garbage-bag-synthetic-images"
datasetSavePath = "./dataset"
datasetClasses = ["garbage", "paper", "plastic"]

datasetImageClassesPath = 'dataset/Bag Classes/Bag Classes'

garbageBagsClassPath = os.path.join(datasetImageClassesPath, "Garbage Bag Images")
paperBagsClassPath = os.path.join(datasetImageClassesPath, "Paper Bag Images")
plasticBagsClassPath = os.path.join(datasetImageClassesPath, "Plastic Bag Images")

randomState = 42

datasetTestPart = 0.2

alpha = 0.05
