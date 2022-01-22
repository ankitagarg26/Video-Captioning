import argparse
import feature_extraction
import numpy as np
import utils

from model import EncoderDecoder
from data_augmentation import augment_data

MAX_ENCODER_SEQ_LENGTH = 30
MAX_DECODER_SEQ_LENGTH = 3
FEATURE_VECTOR_LENGTH = 2048
OUTPUT_SEQ_LENGTH = 117  # equal to number of words in the vocab


# extracting features from video frames using a deep learning CNN model
def prepare_train_data(train_base_dir, augment=0):
    video_ids = utils.get_video_ids(train_base_dir)
    annotations = utils.get_given_annotations()

    input_videos = []
    target_texts = []
    for input_video in video_ids:
        target_text = annotations[id]
        input_videos.append(input_video)
        target_texts.append(target_text)

    encoder_input_data = np.zeros((len(input_videos), MAX_ENCODER_SEQ_LENGTH, FEATURE_VECTOR_LENGTH), dtype='float32')
    decoder_input_data = np.zeros((len(input_videos), MAX_DECODER_SEQ_LENGTH, OUTPUT_SEQ_LENGTH), dtype='float32')
    decoder_target_data = np.zeros((len(input_videos), MAX_DECODER_SEQ_LENGTH, OUTPUT_SEQ_LENGTH), dtype='float32')

    feature_model = feature_extraction.resnet_feature_extractor()
    for i, (input_video, target_text) in enumerate(zip(input_videos, target_texts)):
        video = utils.read_videos([input_video], train_base_dir)
        if augment == 1:
            video = augment_data(video[0])

        encoder_input_data[i] = feature_model.predict([feature_extraction.preprocess_videos(video)])
        for t, annotation in enumerate(target_text):
            decoder_target_data[i, t, annotation] = 1
            if t > 0:
                decoder_input_data[i, t] = decoder_target_data[i, t - 1]

    return encoder_input_data, decoder_input_data, decoder_target_data


def train_model(train_base_dir, batch_size, epochs):
    encoder_decoder = EncoderDecoder(OUTPUT_SEQ_LENGTH, FEATURE_VECTOR_LENGTH)
    model = encoder_decoder.load_training_model()

    encoder_input_data, decoder_input_data, decoder_target_data = prepare_train_data(train_base_dir)
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs)

    # training the model with augmented data
    for j in range(0, 10):
        encoder_input_data, decoder_input_data, decoder_target_data = prepare_train_data(train_base_dir, augment=1)
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs / 2, validation_split=0.2)

    model.save('model.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='directory path for training data', default='./data/train/train/')
    parser.add_argument('--batch_size', type=int, help='batch size for training', default=32)
    parser.add_argument('--epochs', type=int, help='number of epochs for training', default=10)
    args = parser.parse_args()

    train_model(args.train_data, args.batch_size, args.epochs)
