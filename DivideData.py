import pandas as pd
import numpy as np
#数据分类
data = pd.read_csv('fer2013.csv')
train_data = data[data['Usage'] == 'Training']
validate_data = data[data['Usage'] == 'PublicTest']
test_data = data[data['Usage'] == 'PrivateTest']
dataRootPath = "./data//"
train_data.to_csv(dataRootPath + "Train.csv")
validate_data.to_csv(dataRootPath + "Val.csv")
test_data.to_csv(dataRootPath + "Test.csv")
print("Divided Dataset")