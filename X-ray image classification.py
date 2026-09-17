from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from PIL import Image #for image
import numpy as np
import os #for file

#load
train = r"C:\Users\Johnnel\Desktop\TIP folder\Others\free data\archive\chest_xray\train"
test = r"C:\Users\Johnnel\Desktop\TIP folder\Others\free data\archive\chest_xray\test"

X_train = []
y_train = []
X_test = []
y_test = []

for label in ["NORMAL", "PNEUMONIA"]:
    class_folder = os.path.join(train, label)
    for file in os.listdir(class_folder):
        img = Image.open(os.path.join(class_folder, file)).convert("L").resize((64, 64))
        X_train.append(np.array(img).flatten())
        y_train.append(label)

for label in ["NORMAL", "PNEUMONIA"]:
    class_folder = os.path.join(test, label)
    for file in os.listdir(class_folder):
        img = Image.open(os.path.join(class_folder, file)).convert("L").resize((64, 64))
        X_test.append(np.array(img).flatten())
        y_test.append(label)
#model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]  # probability of PNEUMONIA (the "positive" class)
y_test_binary = [1 if label == "PNEUMONIA" else 0 for label in y_test]

#eval
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

ROC_AUC=roc_auc_score(y_test_binary, rf_proba)
print(f"ROC_AUC:{ROC_AUC:.2f}")


#plotting
fpr, tpr, thresholds = roc_curve(y_test_binary, rf_proba)

plt.plot(fpr, tpr, label=f"ROC_AUC = {ROC_AUC:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")  # reference line for random guessing
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

