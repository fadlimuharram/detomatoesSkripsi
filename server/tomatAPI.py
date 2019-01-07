#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:15:13 2018

@author: fadlimuharram
"""

# import the necessary packages
from time import time
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Conv2D

from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LeakyReLU
import numpy as np
from keras.preprocessing import image
import flask
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers.normalization import BatchNormalization



app = flask.Flask(__name__)
klasifikasi = None
train_set = None
test_set = None
datanya = None
jumlahKelas = None

'''development atau production atau initial'''

MODENYA = None 
productionEpochnya = None

print('Input Dengan Menggunakan Model Yang Telah Tersedia')
print('[0] Tidak')
print('[1] Ya')
isLoadedDariModel = int(input('pilihan: '))
print(isLoadedDariModel, type(isLoadedDariModel))
print(isLoadedDariModel != 0)
if isLoadedDariModel != 1 and isLoadedDariModel != 0 :
    raise ValueError('Error: Mohon Pilih 0 atau 1')

if isLoadedDariModel == 1:
    isLoadedDariModel = True
    print('Masukan Jumlah Epoch Sebelumnya')
    productionEpochnya = int(input('jumlah: '))
elif isLoadedDariModel == 0:
    isLoadedDariModel = False
    print('Pilih Mode Training')
    print('[0] initial')
    print('[1] development')
    print('[2] Production')
    MODENYA = int(input('pilihan: '))
    if MODENYA == 0:
        MODENYA = 'initial'
    elif MODENYA == 1:
        MODENYA = 'development'
    elif MODENYA == 2:
        MODENYA = 'production'
    else:
        raise ValueError('Error: pilih mode 0 - 2')
    
    if MODENYA == 'development' or MODENYA == 'production':
        print('Pilih Jumlah Epoch Yang Di Inginkan')
        productionEpochnya = int(input('jumlah: '))
        
else:
    raise ValueError('Error: pilih 0 atau 1 saja')


#isLoadedDariModel = True
#productionEpochnya = 5

IPNYA = '192.168.43.70'
PORTNYA = 5050


LOKASI_TRAINING = '/Users/fadlimuharram/Documents/cnn/tomat/segmentTambahan/training_set'
LOKASI_TESTING = '/Users/fadlimuharram/Documents/cnn/tomat/ini/testing_set'

#LOKASI_TRAINING = '/Users/fadlimuharram/Documents/cnn/tomat/segmentTambahan/training_set'
#LOKASI_TESTING = '/Users/fadlimuharram/Documents/cnn/tomat/segmentTambahan/testing_set'

LOKASI_UPLOAD = 'upload'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

PENYAKIT_TANAMAN = {
        0: {
                "nama":"Bercak Bakteri",
                "penyebab":"Xanthomonas gardneri",
                "penyebaran":"Dunia",
                "gejala":"Tanaman yang terinfeksi penyakit tomat ini batangnya menyerupai kerak yang panjang dengan warna keabu-abuan Sedangkan daun yang terserang penyakit tomat ini akan mengalami klorosis dan rontok atau gugur. Lain halnya jika yang terserang adalah buah, jika pada buah yang terserang bakteri maka akan muncul bercak berair yang kemudian akan berubah menjadi bercak bergabus.",
                "penangan":[
                        "Merotasi tanaman dengan tanaman yang familinya berbeda.",
                        "Menanam benih dari biji tomat yang sehat.",
                        "Menanam bibit tomat yang memiliki ketahanan terhadap hama dan penyakit.",
                        "Tanaman yang sakit sebaiknya dicabut dan dibakar di areal yang jauh dari lahan penanaman.",
                        "Tanaman yang mati sebaiknya jangan dipendam di lahan penanaman.",
                        "Menyiram tanaman dengan menggunakan air yang bersih dan terbebas dari penyakit."
                        ]
            },
        1: {
                "nama":"Hawar Daun",
                "penyebab":"jamur Phytophthora infestans",
                "penyebaran":"Dunia",
                "gejala":"Cendawan dapat menyerang semua tingkat pertumbuhan tanaman. Bercak hitam kecoklatan atau keunguan mulai timbul pada anak daun, tangkai, atau batang.  Pada keadaan kelembaban tinggi, bercak akan cepat meluas, sehingga dapat menyebabkan kematian tanaman. Pada keadaan tersebut bagian paling luar bercak berwarna kuning pucat beralih ke bagian yang berwarna hijau. Pada sisi bawah daun, fruktifikasi cendawan yang berwarna putih seperti beledu tampak pada daerah peralihan antara pucat dan ungu. Gejala bercak pada buah tomat berwarna hijau kelabu kebasahan, meluas menjadi bercak yang bentuk dan besarnya tidak tertentu. Pada buah muda bercak berwarna coklat tua, agak keras dan berkerut. Bercak mempunyai batas yang cukup jelas dan tetap hijau pada waktu bagian yang sehat matang. Kadang-kadang bercak mempunyai cincin-cincin.",
                "penangan":[
                        "Dengan mengatur waktu tanam dimusim kemarau",
                        "Pergiliran (rotasi) tanaman dengan jenis yang bukan inang penyakit hawar daun."]
            },
        2: {
                "nama":"Bercak Daun Septoria",
                "penyebab":"Septoria Lycopersici",
                "penyebaran":"Dunia",
                "gejala":"Timbul bercak kecil bulat dan berair dikedua permukaan daun bagian bawah. Bercak yang timbul berwarna coklat muda yang berubah menjadi kelabu dan tepi berwarna kehitaman. Garis tengah pada bercak mencapai ± 2 mm dan serangan terhebat penyakit tomat ini dapat menyebabkan daun menggulung, kering dan rontok",
                "penangan":[
                        "Gulma beserta tanaman tomat yang mati dibersihkan dari area lahan kemudian dibakar (jangan dipendam dalam tanah).",
                        "Lakukan rotasi tanaman, tanamlah tanaman yang berbeda familinya supaya penyakit tidak menyebar.",
                        "Tanamlah bibit tomat yang tahan hama penyakit (resisten)",
                        "Semprot tanaman dengan menggunakan fungisida."
                        ]
            },
        3: {
                "nama":"Tungau",
                "penyebab":"Polyphagotarsonemus latus",
                "gejala":"Tungau menghisap cairan pada daun muda sehingga tanaman akan menjadi nekrotik, kaku, dan keriting. Pertumbuhan tanaman menjadi terhambat. Gejala serangan sangat cepat terlihat, yaitu pada hari ke 8 – 10 tanaman  pucuk tanaman menjadi berwarna coklat.",
                "penangan":[
                        "Pengendalian dengan agens hayati menggunakan MOSA BN dengan bahan aktif jamur Beauveria bassiana. Penyemprotan dilakukan sore hari dengan dosis 2,5 gram per liter atau 30 gram per tangki isi 14 liter air. Saat dicampur dengan air ditambahkan gula  2 sendok atau Molase.",
                        "Penyemprotan dengan insekisida Nabati/Botanik dengan bahan aktif Metilanol 100 g/l. Dosis penyemrotan 1 – 1,5 ml/l atau 10 ml /tangki 14 liter."
                        ]
            }
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
           
def load_model_klasifikasi():
    global klasifikasi, train_set, test_set, datanya, kelasnya, LOKASI_TRAINING, LOKASI_TESTING
    global MODENYA, productionEpochnya, isLoadedDariModel

    klasifikasi = Sequential()
    
    
  
    
    
    klasifikasi.add(Convolution2D(12,    # jumlah filter layers
                        3,    # y dimensi kernel
                        3,    # x dimensi kernel
                        input_shape=(64, 64, 3)))
    
    klasifikasi.add(Activation('relu'))
    klasifikasi.add(MaxPooling2D(pool_size=(2,2)))
    klasifikasi.add(Convolution2D(25,
                            3,
                            3))

    klasifikasi.add(Activation('relu'))
    klasifikasi.add(MaxPooling2D(pool_size=(2,2)))
    
    
    
    
    #Flatten
    klasifikasi.add(Flatten())
    klasifikasi.add(Dense(675, activation = 'relu'))
    klasifikasi.add(Dropout(0.2))
    klasifikasi.add(Dense(507, activation = 'relu'))
    klasifikasi.add(Dropout(0.2))
    
  
    '''
    # 1st Convolutional Layer
    klasifikasi.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
     strides=(4,4), padding='valid'))
    klasifikasi.add(Activation('relu'))
    # Pooling 
    klasifikasi.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    klasifikasi.add(BatchNormalization())
    
    # 2nd Convolutional Layer
    klasifikasi.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    klasifikasi.add(Activation('relu'))
    # Pooling
    klasifikasi.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    klasifikasi.add(BatchNormalization())
    
    # 3rd Convolutional Layer
    klasifikasi.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    klasifikasi.add(Activation('relu'))
    # Batch Normalisation
    klasifikasi.add(BatchNormalization())
    
    # 4th Convolutional Layer
    klasifikasi.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    klasifikasi.add(Activation('relu'))
    # Batch Normalisation
    klasifikasi.add(BatchNormalization())
    
    # 5th Convolutional Layer
    klasifikasi.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    klasifikasi.add(Activation('relu'))
    # Pooling
    klasifikasi.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    klasifikasi.add(BatchNormalization())
    
    # Passing it to a dense layer
    klasifikasi.add(Flatten())
    # 1st Dense Layer
    klasifikasi.add(Dense(4096, input_shape=(224*224*3,)))
    klasifikasi.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    klasifikasi.add(Dropout(0.4))
    # Batch Normalisation
    klasifikasi.add(BatchNormalization())
    
    # 2nd Dense Layer
    klasifikasi.add(Dense(4096))
    klasifikasi.add(Activation('relu'))
    # Add Dropout
    klasifikasi.add(Dropout(0.4))
    # Batch Normalisation
    klasifikasi.add(BatchNormalization())
    
    # 3rd Dense Layer
    klasifikasi.add(Dense(1000))
    klasifikasi.add(Activation('relu'))
    # Add Dropout
    klasifikasi.add(Dropout(0.4))
    # Batch Normalisation
    klasifikasi.add(BatchNormalization())

    '''
    
    klasifikasi.add(Dense(jumlahKelas, activation='softmax',init='he_normal'))
    print("Full Connection Between Hidden Layers and Output Layers Completed")

    
    
    
    
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
    
    if isLoadedDariModel == True:
        namaFilenya = "modelKlasifikasi" + str(productionEpochnya) +".h5"
        if os.path.exists(namaFilenya) :
            klasifikasi = load_model(namaFilenya)
            datanya = klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
        else:
            raise ValueError('Error: File Tidak Ada Harap Lakukan Training Terlebih Dahulu Sebelum Menggunakan Model')
    else:
        # kompile cnn
        klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
        print(klasifikasi.summary())
        print("Compiling Initiated")
        if MODENYA == 'development':
            
            datanya = klasifikasi.fit_generator(
                train_set,
                steps_per_epoch=50,
                epochs=productionEpochnya,
                validation_data=test_set,
                validation_steps=30)
            
            klasifikasi.save("modelKlasifikasiinibro" + str(productionEpochnya) +".h5")
            
        elif MODENYA == 'production' :
            
            tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
            
            datanya = klasifikasi.fit_generator(
                train_set,
                steps_per_epoch=hitungGambar(LOKASI_TRAINING),
                epochs=productionEpochnya,
                validation_data=test_set,
                validation_steps=hitungGambar(LOKASI_TESTING),
                callbacks=[tensorboard]
                )
            klasifikasi.save("modelKlasifikasilima" + str(productionEpochnya) +".h5")
            
        elif MODENYA == 'initial' :
            
            datanya = klasifikasi.fit_generator(
                train_set,
                steps_per_epoch=5,
                epochs=1,
                validation_data=test_set,
                validation_steps=2)
        gambarHasilLatih()
        
    klasifikasi._make_predict_function()
    print("Compiling Completed")


def gambarHasilLatih():
    global datanya
    # Plot training & validation accuracy
    plt.plot(datanya.history['acc'])
    plt.plot(datanya.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss
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
        
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        
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
            result = klasifikasi.predict(test_image).tolist()
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
                    hasil["proba" + str(loop)] = PENYAKIT_TANAMAN[val]
                    hasil["proba" + str(loop)]['probability'] = proba
            
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
    load_model_klasifikasi()
    app.run(host=IPNYA, port=PORTNYA,debug=True)