import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

data = []
labels = []
classes = 43
cur_path = os.getcwd()

classs = {
    1:"Speed limit (20km/h)", 2:"Speed limit (30km/h)", 3:"Speed limit (50km/h)",
    4:"Speed limit (60km/h)", 5:"Speed limit (70km/h)", 6:"Speed limit (80km/h)",
    7:"End of speed limit (80km/h)", 8:"Speed limit (100km/h)", 9:"Speed limit (120km/h)",
    10:"No passing", 11:"No passing veh over 3.5 tons", 12:"Right-of-way at intersection",
    13:"Priority road", 14:"Yield", 15:"Stop", 16:"No vehicles",
    17:"Veh > 3.5 tons prohibited", 18:"No entry", 19:"General caution",
    20:"Dangerous curve left", 21:"Dangerous curve right", 22:"Double curve",
    23:"Bumpy road", 24:"Slippery road", 25:"Road narrows on the right",
    26:"Road work", 27:"Traffic signals", 28:"Pedestrians",
    29:"Children crossing", 30:"Bicycles crossing", 31:"Beware of ice/snow",
    32:"Wild animals crossing", 33:"End speed + passing limits", 34:"Turn right ahead",
    35:"Turn left ahead", 36:"Ahead only", 37:"Go straight or right",
    38:"Go straight or left", 39:"Keep right", 40:"Keep left",
    41:"Roundabout mandatory", 42:"End of no passing", 43:"End no passing veh > 3.5 tons"
}

print("Obtaining Images & its Labels..............")
for i in range(classes):
    path = os.path.join(cur_path, 'dataset/train/', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '/' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
            print(f"{a} Loaded")
        except:
            print("Error loading image")

print("Dataset Loaded")

data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")

        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        self.label_2.setFont(font)

        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 200, 16))

        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)

        # Load model if exists
        if os.path.exists("my_model.h5"):
            self.model = load_model("my_model.h5")
        else:
            self.model = None

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle("Road Sign Recognition")
        self.BrowseImage.setText("Browse Image")
        self.label_2.setText("           ROAD SIGN RECOGNITION")
        self.Classify.setText("Predict")
        self.label.setText("Recognized Sign")
        self.Training.setText("Training")

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if fileName:
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(),
                                   QtCore.Qt.KeepAspectRatio)
            self.imageLbl.setPixmap(pixmap)

    def classifyFunction(self):
        if not hasattr(self, "file"):
            self.textEdit.setText("Please select an image first.")
            return

        if self.model is None:
            self.textEdit.setText("Model not trained or loaded.")
            return

        image = Image.open(self.file)
        image = image.resize((30, 30))
        image = np.array(image)
        image = image.reshape(1, 30, 30, 3)

        prediction = self.model.predict(image)
        result = int(np.argmax(prediction, axis=1)[0])

        sign = classs[result + 1]
        print(sign)
        self.textEdit.setText(sign)

    def trainingFunction(self):
        self.textEdit.setText("Training under process...")

        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=X_train.shape[1:]))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(43, activation='softmax'))
        print("Initialized model")

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(X_train, y_train, batch_size=32, epochs=10,
                            validation_data=(X_test, y_test))

        model.save("my_model.h5")
        self.model = model

        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('Accuracy.png')

        plt.figure(1)
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('Loss.png')

        self.textEdit.setText("Saved Model & Graphs to disk")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Check if model already exists
    if os.path.exists("my_model.h5"):
        model = load_model("my_model.h5")
        # Load class indices from dataset
        train_gen, _ = prepare_data()
        class_indices = train_gen.class_indices
        print("🟢 Model loaded — skipping training")
    else:
        # Train model if not present
        model, class_indices = train_model()
        print("🟢 Model trained and saved")

