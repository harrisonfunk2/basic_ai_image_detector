from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import resnet_embedding 

x_train = resnet_embedding.X_train
y_train = resnet_embedding.y_train
x_test = resnet_embedding.X_test
y_test = resnet_embedding.y_test

model = LogisticRegression(solver='liblinear', random_state=0, max_iter=2000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("accuracy:", model.score(x_test, y_test))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["real", "ai"]))

probs = model.predict_proba(x_test)[:, 1]  # P(ai)
print(probs[:10])