 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:15:13 2018

@author: fadlimuharram
"""

# import the necessary packages
from time import time
from keras.models import Sequential
from keras.layers import Conv2D, Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import flask
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
klasifikasi = None
train_set = None
test_set = None
datanya = None
jumlahKelas = None

'''development atau production atau initial'''

MODENYA = 'production' 

IPNYA = '192.168.43.70'
PORTNYA = 5050


LOKASI_TRAINING = '/Users/fadlimuharram/Documents/cnn/tomat/segmentTambahan/training_set'
LOKASI_TESTING = '/Users/fadlimuharram/Documents/cnn/tomat/segmentTambahan/testing_set'
LOKASI_UPLOAD = 'upload'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

PENYAKIT_TANAMAN = {
        0: {"nama":"Bacterial Spot","gejala":"abc gejala 0","penangan":"www 0"},
        1: {"nama":"Late Blight","gejala":"abc gejala 1","penangan":"www 1"},
        2: {"nama":"Septoria Leaf Spot","gejala":"abc gejala 2","penangan":"www 2"},
        3: {"nama":"Spider Mites","gejala":"abc gejala 3","penangan":"www 3"}
        }

print(PENYAKIT_TANAMAN[0])
def hitungGambar(path):
    count = 0
    for filename in os.listdir(path):
        if filename != '.DS_Store':
            count = count + len(os.listdir(path+'/'+filename))
    return count

def hitunKelas():
    global LOKASI_TRAINING, LOKASI_TESTING, PENYAKIT_TANAMAN
    kelasTraining = 0
    kelasTesting = 0
    
    for filename in os.listdir(LOKASI_TRAINING):
        if filename != '.DS_Store':
            kelasTraining = kelasTraining + 1
            
    for filename in os.listdir(LOKASI_TESTING):
        if filename != '.DS_Store':
            kelasTesting = kelasTesting + 1
            
    if kelasTesting == kelasTraining and kelasTraining == len(PENYAKIT_TANAMAN) and kelasTesting == len(PENYAKIT_TANAMAN):
        return kelasTraining
    else:
        raise ValueError('Error: Kelas Training tidak sama dengan Kelas Testing')
        






app.config['UPLOAD_FOLDER'] = LOKASI_UPLOAD
app.config['STATIC_FOLDER'] = LOKASI_UPLOAD
jumlahKelas = hitunKelas()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def load_model():
    global klasifikasi, train_set, test_set, datanya, kelasnya, LOKASI_TRAINING, LOKASI_TESTING
    global MODENYA
    # initialising the cnn
    klasifikasi = Sequential()
    
    
    # 1st Convolutional Layer
    klasifikasi.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(64, 64, 3)))
    klasifikasi.add(Activation('relu'))
    # Max Pooling
    klasifikasi.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 2nd Convolutional Layer
    klasifikasi.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    klasifikasi.add(Activation('relu'))
    # Max Pooling
    klasifikasi.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 3rd Convolutional Layer
    klasifikasi.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    klasifikasi.add(Activation('relu'))
    
    # 4th Convolutional Layer
    klasifikasi.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    klasifikasi.add(Activation('relu'))
    
    # 5th Convolutional Layer
    klasifikasi.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    klasifikasi.add(Activation('relu'))
    # Max Pooling
    klasifikasi.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # Passing it to a Fully Connected layer
    klasifikasi.add(Flatten())
    # 1st Fully Connected Layer
    klasifikasi.add(Dense(4096, input_shape=(224*224*3,)))
    klasifikasi.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    klasifikasi.add(Dropout(0.4))
    
    # 2nd Fully Connected Layer
    klasifikasi.add(Dense(4096))
    klasifikasi.add(Activation('relu'))
    # Add Dropout
    klasifikasi.add(Dropout(0.4))
    
    # 3rd Fully Connected Layer
    klasifikasi.add(Dense(1000))
    klasifikasi.add(Activation('relu'))
    # Add Dropout
    klasifikasi.add(Dropout(0.4))
    
    # Output Layer
    klasifikasi.add(Dense(17))
    
    klasifikasi.add(Dense(jumlahKelas, activation='softmax'))
    print("Full Connection Between Hidden Layers and Output Layers Completed")

    
    # kompile cnn
    klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
    print("Compiling Initiated")
    
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_set = train_datagen.flow_from_directory(
            LOKASI_TRAINING,
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(
            LOKASI_TESTING,
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical')
    
    
    
    if MODENYA == 'development':
        
        datanya = klasifikasi.fit_generator(
            train_set,
            steps_per_epoch=50,
            epochs=5,
            validation_data=test_set,
            validation_steps=30)
        
    elif MODENYA == 'production' :
        
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        
        datanya = klasifikasi.fit_generator(
            train_set,
            steps_per_epoch=hitungGambar(LOKASI_TRAINING),
            epochs=5,
            validation_data=test_set,
            validation_steps=hitungGambar(LOKASI_TESTING),
            callbacks=[tensorboard])
        
    elif MODENYA == 'initial' :
        
        datanya = klasifikasi.fit_generator(
            train_set,
            steps_per_epoch=5,
            epochs=1,
            validation_data=test_set,
            validation_steps=2)
        
    klasifikasi._make_predict_function()
    gambarHasilLatih()
    print("Compiling Completed")


def gambarHasilLatih():
    global datanya
    # Plot training & validation accuracy values
    plt.plot(datanya.history['acc'])
    plt.plot(datanya.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(datanya.history['loss'])
    plt.plot(datanya.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

        
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    global train_set, klasifikasi, IPNYA, PORTNYA, LOKASI_UPLOAD, PENYAKIT_TANAMAN
    print('-------------')
    print(request.method)
    print(request.files)
    print('-------------')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save('static/' + os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)
            
            lokasiTest = LOKASI_UPLOAD + '/' + filename
          
            test_image = image.load_img('static/' + lokasiTest, target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = klasifikasi.predict_proba(test_image).tolist()
            '''result = pd.Series(result).to_json(orient='values')'''
            print(train_set.class_indices)
            '''return redirect(url_for('uploaded_file',filename=filename))'''
            print(result)
            
            hasil = {}
            dataJSON = {}
            allProba = {}
            loop = 0
            
            for cls, val in train_set.class_indices.items():
                '''hasil[cls] = result[0][train_set.class_indices[cls]]'''
                
                proba = result[0][train_set.class_indices[cls]]
                allProba[cls] = proba
                print(proba)
                if (proba > 0.0) and (proba <= 1.0) :
                    print('valnya : ' + str(val))
                    '''hasil.update({'datanya':{PENYAKIT_TANAMAN[val]},'probability':proba})'''
                    hasil["proba-" + str(loop)] = PENYAKIT_TANAMAN[val]
                    hasil["proba-" + str(loop)]['probability'] = proba
            
                    loop = loop + 1
            print(hasil)
            dataJSON['Debug'] = allProba
            dataJSON['penyakit'] = hasil
            dataJSON['uploadURI'] = 'http://' + IPNYA + ':' + str(PORTNYA) + url_for('static',filename=lokasiTest)
            
            return flask.jsonify(dataJSON)
        

    else:
        
        return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method=post enctype='multipart/form-data'>
              <p><input type='file' name='file'>
                 <input type='submit' value='Upload'>
            </form>
            '''
  
        
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host=IPNYA, port=PORTNYA,debug=True)