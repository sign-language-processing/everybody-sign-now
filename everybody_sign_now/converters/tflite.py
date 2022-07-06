import tensorflow as tf

model = tf.keras.models.load_model('../../generator_fixed.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

open("Pix2PixGenerator.tflite", "wb").write(tflite_model)

