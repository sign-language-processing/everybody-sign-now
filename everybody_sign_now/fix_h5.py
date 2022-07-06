from everybody_sign_now.model import generator
import tensorflow as tf

generator.load_weights("generator.h5")

problematic_weight = generator.weights[35]
problematic_weight.assign(tf.zeros(problematic_weight.shape))

generator.save("generator_fixed.h5")
