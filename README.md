# Video-Captioning

A machine Learning project to generate captions for video frames indicating the relationship between the objects in the video.

## Approach
In our framework we use a sequence-to-sequence model to perform video visual  relationship predictions where the input is a sequence of video frames and the output is a relation triplet < object1 − relationship − object2 > representing the videos. We extend the sequence-to-sequence modelling approach to an input of sequence of video frames.

![image](https://user-images.githubusercontent.com/79797476/150630904-3bee867a-a402-4949-9cef-0b50fa0cc7f8.png)

Figure: Bidirectional LSTM layer (coloured red) encodes visual feature inputs, and the LSTM layer (coloured green) decodes the features into a sequence of words.

## Results

![image](https://user-images.githubusercontent.com/79797476/150638324-5ebfedff-b28e-4480-86b4-a0030e197a68.png)



## Python Dependencies
1. Pandas
2. Keras 
3. Tensorflow
4. Numpy
5. albumenations
6. Pillow

## Procedure 

### Training 

For training the model, run the script train.py.

      python train.py
      
For training on your own dataset:
Save your data in a directory (for the format check the data folder). 
Update the json files.

  1. object1_object2.json: 
     It contains a dictionary for each object, with object labels as keys and ids as values.

  2. relationship.json:
     It contains a dictionary for each relationship, with relationship labels as keys and ids as values.

  3. training_annotations.json:
     It contains a dictionary for each video in the training data, with video ids as keys and a list of <object1, relationship, object2> as values.
       
While running the script provide your directory path.

      python eval.py --train_data <directory_path>
      
 
### Testing
For testing the model or making predictions on your own dataset, run the script eval.py.

      python eval.py --test_data <directory_path>

Result will be saved to a csv file 'test_data_predictions.csv'.
      
      

