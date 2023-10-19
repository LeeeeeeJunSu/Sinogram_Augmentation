import Engine
from tifffile import imwrite
import os

trainImagePath = os.getcwd() + "/Data/Train/"
testImagePath = os.getcwd() + "/Data/Test/"
resultPath = os.getcwd() + "/Data/Result/"
ModelPath = os.getcwd() + "/Data/Model/"

e = Engine.engine()
e.loadModel(ModelPath)
#e.train(trainImagePath, epochs = 10, batchSize = 720, learningRate = 0.0002, betas = (0.5,0.999), debugStep = 10, delete_zero = True)
#e.saveModel(ModelPath)
resultList = e.predict(testImagePath)
for image in resultList:
    imwrite(f"{resultPath}{image[0]}", image[1])
