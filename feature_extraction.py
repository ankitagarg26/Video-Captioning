import numpy as np
from keras.applications import ResNet152
from keras.applications.resnet import preprocess_input
from keras.layers import TimeDistributed
from keras.preprocessing import image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def preprocess_videos(read_frames):
    frame_data = []
    for i in range(0, 30):
        img = read_frames[i]
        img = image.smart_resize(img, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        frame_data.append(x.reshape(224, 224, 3))

    return np.expand_dims(np.array(frame_data), axis=0)


# Extracting the output of the second last layer of the pre-trained resnet152 model for each video frame
def resnet_feature_extractor():
    video_input = Input(shape=(30, 224, 224, 3))
    model = ResNet152(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    for layer in model.layers:
        layer.trainable = False

    encoded_frame_sequence = TimeDistributed(model)(video_input)

    feature_extract_model = Model(inputs=video_input, outputs=encoded_frame_sequence)

    return feature_extract_model
