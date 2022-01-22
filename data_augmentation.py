import albumentations as A
import numpy as np

transform = A.ReplayCompose([
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=(-15, 15), p=0.5),
    A.Downscale(p=0.2),
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.2))
])

def augment_data(frames):
    aug_frames = []
    for i in range(0, 30):
        if i == 0:
            data = transform(image=frames[i])
            img = data['image']
        else:
            replay_image = A.ReplayCompose.replay(data['replay'], image=frames[i])
            img = replay_image['image']

        aug_frames.append(img)

    return np.array(aug_frames)