import json
import os
import numpy as np

from PIL import Image
from numpy import asarray


def read_videos(video_ids, base_dir):
    frames = []
    image_files = []
    for video_id in video_ids:
        for img_file in os.listdir(base_dir + "/" + video_id):
            image_files.append(img_file)
        image_files.sort()
        frame_data = []
        for i in range(0, 30):
            img_file = image_files[i]
            img_path = base_dir + "/" + video_id + "/" + img_file
            img = Image.open(img_path)
            numpydata = asarray(img)
            frame_data.append(numpydata)

        frames.append(np.array(frame_data))

    return np.array(frames)

def get_video_ids(base_dir):
    video_ids = []
    for ids in os.listdir(base_dir):
        video_ids.append(ids)
    return video_ids

def get_idx_object_map():
    with open('./object1_object2.json') as f:
        object_idx_map = json.load(f)
    idx_object_map = {}
    for key in object_idx_map.keys():
        idx_object_map[object_idx_map[key]] = key

    return idx_object_map


def get_idx_relationship_map():
    with open('./relationship.json') as f:
        relationship_idx_map = json.load(f)
    idx_relation_map = {}
    for key in relationship_idx_map.keys():
        idx_relation_map[relationship_idx_map[key]] = key

    return idx_relation_map


def get_given_annotations():
    with open('../training_annotations.json') as f:
        given_annotation = json.load(f)
    return given_annotation
