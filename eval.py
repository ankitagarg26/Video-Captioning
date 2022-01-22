import numpy as np
import pandas as pd
import argparse
import feature_extraction
import utils

from model import EncoderDecoder

FEATURE_VECTOR_LENGTH = 2048
OUTPUT_SEQ_LENGTH = 117  # equal to number of words in the vocab


def encode_decode_sequence(encoder_model, decoder_model, input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, OUTPUT_SEQ_LENGTH))
    result = np.zeros((3, 3, 117))
    for i in range(0, 3):
        decoder_input = [target_seq] + states_value
        output_tokens, h, c = decoder_model.predict(decoder_input)

        result[i] = output_tokens[0][0]
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        target_seq = np.zeros((1, 1, OUTPUT_SEQ_LENGTH))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return result


def prepare_test_data(test_base_dir, video_ids):
    input_sequences = {}
    feature_model = feature_extraction.resnet_feature_extractor()

    for video_id in video_ids:
        video = utils.read_videos([video_id], test_base_dir)
        input_sequences[video_id] = feature_model.predict([feature_extraction.preprocess_videos(video[0])])

    return input_sequences


def get_valid_predictions(prediction, is_relationship):
    valid_pred = np.zeros_like(prediction)
    relationship_idx = utils.get_idx_relationship_map()
    object_idx = utils.get_idx_object_map()

    # removing ids which are not valid like for objects removing ids that corresponds to relationship
    if is_relationship == 1:
        for idx in relationship_idx:
            valid_pred[idx] = prediction[idx]
    else:
        for idx in object_idx:
            valid_pred[idx] = prediction[idx]

    # taking the top scoring prediction value
    result = valid_pred.argsort()[-1:][::-1]

    return result


def predict(test_base_dir, model_path):
    test_video_ids = utils.get_video_ids(test_base_dir)
    input_sequences = prepare_test_data(test_base_dir, test_video_ids)

    encoder_decoder = EncoderDecoder(OUTPUT_SEQ_LENGTH, FEATURE_VECTOR_LENGTH)
    encoder_decoder.load_weights(model_path)
    encoder_model = encoder_decoder.load_test_encoder()
    decoder_model = encoder_decoder.load_test_decoder()

    result = []
    for video_id in test_video_ids:
        prediction = encode_decode_sequence(encoder_model, decoder_model, input_sequences[video_id])
        row = {}
        ids = []
        for i in range(0, 3):
            row['VIDEO_ID'] = video_id
            ids.append(get_valid_predictions(prediction[i], i))
            result.append(row)

    df = pd.DataFrame(result)
    df.to_csv('test_data_predictions.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, help='directory path for test data', default='./data/test/test/')
    parser.add_argument('--model_path', type=int, help='path for the saved mdoel', default='model.h5')
    args = parser.parse_args()

    predict(args.train_data, args.model_path)
