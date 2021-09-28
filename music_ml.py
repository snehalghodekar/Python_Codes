import pandas as pd
from sklearn.tree import DecisionTreeClassifier


music_data = pd.read_csv('xyz.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']


model = DecisionTreeClassifier()
model.fit(x,y)
predictions = model.predict([[21,1],[22,0]])
