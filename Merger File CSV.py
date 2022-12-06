from pandas.core.indexes.range import RangeIndex
import pandas as pd
import numpy as np
import os

PC = 10
width = 100
formatPrint = "PC{col}.{row}"
columnName = ['Label Objek']
for i in range(PC):
 for j in range(width):
  columnName.append(formatPrint.format(col = i+1, row = j+1))
dataframe = pd.DataFrame(columns=columnName)

directory = 'csv/'
for filename in os.listdir('csv'):
 if filename.endswith(".csv"):
  df = pd.read_csv(directory + filename, float_precision='round_trip')
  df.pop('Unnamed: 0')
  array = df.transpose().to_numpy().flatten()
  label = os.path.splitext(filename)[0]
  data = [label]
  dd = array.tolist()
  data = data + dd
  dataframe.loc[len(dataframe)] = data

dataframe.to_excel('dataset.xlsx')