import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def train_xgb(data, cols_to_drop, n_estimators):
    """
    Trains a XGB model
    :param data: data to train / validate
    :param cols_to_drop: which columns to drop
    :param n_estimators: number of trees
    :return: xgb model, label encoder
    """

    X = data.drop(columns=cols_to_drop)
    y = data['activity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # encode label
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # train the model
    xgb_classifier = xgb.XGBRFClassifier(n_estimators=n_estimators, random_state=42)
    xgb_classifier.fit(X_train, y_train_encoded)

    # make predictions
    y_pred_proba = xgb_classifier.predict_proba(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    loss = log_loss(y_test, y_pred_proba)

    print("Accuracy:", accuracy)
    print("Log Loss:", loss)
    return xgb_classifier, label_encoder
