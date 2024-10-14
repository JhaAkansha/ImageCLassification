#from fastapi import FastAPI, File, UploadFile
# import keras
# from PIL import Image
# import io
# import tensorflow as tf

# app = FastAPI()


# @app.post("/classify/cats-and-dogs")
# async def root(file: UploadFile):
#     contents = await file.read()
#     target_size = (180,180)
    
#     #print(type(contents))
#     img = Image.open(io.BytesIO(contents))
#     #print(type(img))
#     img = img.convert("RGB")
#     img = img.resize(target_size, Image.NEAREST)
#     model = keras.models.load_model("catsvsdogsmodel.keras")
#     img = keras.utils.img_to_array(img)
#     #print(type(img))
#     img = tf.expand_dims(img, 0)  # Create batch axis
#     predictions = model.predict(img)
#     score = float(tf.sigmoid(predictions[0][0]))
#     print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
#     return {"filename": file.filename}

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import keras
from PIL import Image
import io
import tensorflow as tf


target_size = (180, 180)
print(keras.__version__)
print(tf.__version__)
# model = keras.models.load_model("catsvsdogsmodel.keras")
model = keras.models.load_model("catsvsdogsmodel\model.weights.h5")
app = FastAPI()


@app.post("/classify/cats-and-dogs")
async def root(file: UploadFile = File(...)):
    contents = await file.read()
    
    # prepare image
    img = Image.open(io.BytesIO(contents))
    img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    
    # make predictions
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    
    # calculate score
    score = float(tf.sigmoid(predictions[0][0]))
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
    return {"probabilities": [{"cat": 100 * (1 - score)}, {"dog": 100 * score}]}