rm -r web_model
rm web_model.zip

tensorflowjs_converter --quantize_float16 --weight_shard_size_bytes 8388608 --input_format=keras generator.h5 web_model

zip web_model.zip web_model/*