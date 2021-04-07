import tensorflow as tf
import segmentation_models as sm


ip = tf.keras.layers.Input(shape = (416,256,3))
pre = tf.keras.applications.mobilenet.preprocess_input(ip)
bm = sm.Unet('mobilenet',input_shape =(416,256,3), encoder_weights ="imagenet",
             activation='linear',
             decoder_use_batchnorm = True, encoder_freeze = True)(pre)
model1 = tf.keras.models.Model(ip, bm)
model1.trainable = True
model1.compile(loss ='mse', optimizer='adam')
model.load_weights('interp_unet.h5')