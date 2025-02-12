import csv
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def read_from_csv(input_filename="landmarks.csv"):
    with open(input_filename, mode='r') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]
    
    x = np.array([row[:-1] for row in data])
    y = np.array([row[-1] for row in data])
    
    return x, y

x, y = read_from_csv()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
precision = precision_score(y_test, y_predict, average='macro')
recall = recall_score(y_test, y_predict, average='macro')
f1 = f1_score(y_test, y_predict, average='macro')

print("Prediction completed!")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

#confusion matrix
cm = confusion_matrix(y_test, y_predict)
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=letters, yticklabels=letters)
plt.show()

#saving model
with open("model.p", "wb") as file:
    pickle.dump({"model": model}, file)
    print("Model saved successfully.")