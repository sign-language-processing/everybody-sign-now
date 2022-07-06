from tensorflow import keras
import coremltools

model = keras.models.load_model('../../generator_fixed.h5')
coreml_model = coremltools.converters.convert(model,
                                              compute_precision=coremltools.precision.FLOAT16,
                                              convert_to="mlprogram",
                                              minimum_deployment_target=coremltools.target.iOS15)

coreml_model.author = 'Amit Moryossef'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Convert a pose estimation image to a human'
coreml_model.input_description['input_1'] = 'A 256x256 pixel Image'

coreml_model.save('Pix2PixGeneratorF16.mlpackage')
