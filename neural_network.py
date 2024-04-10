import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
#from keras_self_attention import SeqSelfAttention
#import seaborn as sns
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from config import SAVE_DIR_PATH
from config import MODEL_DIR_PATH
from keras.layers import GlobalAveragePooling1D, Reshape
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
# '
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score

from keras.models import Model

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score
from keras.callbacks import ReduceLROnPlateau
from xgboost import XGBClassifier
from keras.models import Model

class TrainModel:

    @staticmethod
    def train_neural_network(X, y) -> None:
        """
        This function trains the neural network, adds global average pooling, reduces dimensions, 
        and extracts features for input to SVM.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        print(x_traincnn.shape, x_testcnn.shape)

        model = Sequential()
        model.add(Conv1D(32, 5, padding='same', input_shape=(40, 1)))
        model.add(Activation('swish'))
        model.add(Conv1D(64, 5, padding='same', input_shape=(40, 1)))
        model.add(Activation('swish'))
        model.add(Dropout(0.2))
        model.add(Conv1D(128, 5, padding='same', input_shape=(40, 1)))
        model.add(Activation('swish'))
        model.add(Conv1D(256, 5, padding='same', input_shape=(40, 1)))
        model.add(Activation('swish'))
        model.add(Dropout(0.2))
        model.add(LSTM(256, return_sequences=True))
        model.add(GlobalAveragePooling1D(name='global_average_pooling'))
        model.add(Flatten())
        model.add(Dense(8, activation='softmax'))

        print(model.summary())
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        cnn_history = model.fit(x_traincnn, y_train, batch_size=16, epochs=50, validation_data=(x_testcnn, y_test), callbacks=[early_stopping])
 
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling').output)
        features_train = intermediate_layer_model.predict(x_traincnn)
        features_test = intermediate_layer_model.predict(x_testcnn)

        features_train = features_train[:, :300]
        features_test = features_test[:, :300]

        scaler = StandardScaler()
        features_train_scaled = scaler.fit_transform(features_train)
        features_test_scaled = scaler.transform(features_test)

        return features_train_scaled, features_test_scaled, y_train, y_test

    @staticmethod
    def train_xgboost(X_train, X_test, y_train, y_test):
        xgb_model = XGBClassifier()
        xgb_model.fit(X_train, y_train)
        return xgb_model

    @staticmethod
    def train_svm(X_train, y_train):
        svm_model = SVC(class_weight='balanced')
        svm_model.fit(X_train, y_train)
        return svm_model

    @staticmethod
    def ensemble_predict(predictions_list):
        return np.mean(predictions_list, axis=0)

if __name__ == '__main__':
    print('Training started')
    X = joblib.load(SAVE_DIR_PATH + '\\X.joblib')
    y = joblib.load(SAVE_DIR_PATH + '\\y.joblib')
    
    train_model = TrainModel()
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_model.train_neural_network(X=X, y=y)
    xgb_model = train_model.train_xgboost(X_train_nn, X_test_nn, y_train_nn, y_test_nn)
    svm_model = train_model.train_svm(X_train_nn, y_train_nn)
    
    xgb_predictions = xgb_model.predict_proba(X_test_nn)
    svm_predictions = svm_model.decision_function(X_test_nn)
    ensemble_predictions = train_model.ensemble_predict([xgb_predictions, svm_predictions])
    
    ensemble_accuracy = accuracy_score(y_test_nn, np.argmax(ensemble_predictions, axis=1))
    print(f'Ensemble Accuracy: {ensemble_accuracy}')