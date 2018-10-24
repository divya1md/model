import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np
from keras.models import load_model


baseMapNum = 32
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(baseMapNum, (3,3), padding='same', data_format=None, kernel_regularizer=regularizers.l2(weight_decay), input_shape=(64, 64,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())



model.add(Conv2D(baseMapNum, (3,3), padding='same', data_format=None, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), data_format=None))
model.add(Dropout(0.2))

model.add(Conv2D(2*baseMapNum, (3,3), padding='same', data_format=None, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(2*baseMapNum, (3,3), padding='same', data_format=None, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), data_format=None))
model.add(Dropout(0.3))

model.add(Conv2D(4*baseMapNum, (3,3), padding='same',  data_format=None, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(4*baseMapNum, (3,3), padding='same',  data_format=None, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), data_format=None))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.summary()

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
#datagen.fit(x_train)



training_set = datagen.flow_from_directory('C:\\Users\\Miracle\\Desktop\\car_divya\\car_images\\training',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_set = datagen.flow_from_directory('C:\\Users\\Miracle\\Desktop\\car_divya\\car_images\\testing',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'categorical')




#training
batch_size = 64
#epochs=25
opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(training_set,steps_per_epoch=4000,epochs=10,verbose=1,validation_data=test_set,validation_steps = 1467)
model.save('C:\\Users\\Miracle\\Desktop\\car_divya\\saved_models\\all_specific.h5')

# once the model is saved.. we can load it and donot have to train it each time , we can only run the code from this line with proper input image url


model = load_model('C:\\Users\\Miracle\\Desktop\\car_divya\\saved_models\\all_specific.h5')
# save the model to disk

# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('C:\\Users\\Miracle\\Desktop\\car_divya\\car_images\\training\\glass\\pic_014.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# training_set.class_indices
# print(result)

# #print(result)
# #result.index(np.abs(result).max())
# n=np.argmax(result)



# if n == 0:
    # print("Bumper got damaged")
# elif(n== 1):
    # print("door got damaged")
# elif(n== 2):
    # print("glass got damaged")
# elif(n==3):
    # print("Car not damaged")
# else:
    # print("Trunk got damaged")

# #testing - no kaggle eval
# #scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
# #print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))