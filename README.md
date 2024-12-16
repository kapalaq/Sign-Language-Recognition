# Sign-Language-Recognition
Creating and maintaining model for sign language recognition.

This project consists of OpenCV, MediaPipe and Tensorflow tools with mainly Python Language. I used coordinates of face, body and hands landmarks from each frame and passed them to LSTM in order to process sequence of frames as one. To simplify the first version, a static <i>duration=30 frames</i> were used for each data sample. Since it is barely related to real world data in the future I will change model's capabiities to video duration independence.
* <i>OpenCV</i> was used to process images and getting livestream from Laptop camera.
* MediaPipe's solution <i>Holistic</i> was used for face, body and hands recognition
* Tensorflow was used for composing, training and saving LSTM-based model for sign language recongition

### My project has following objects:
1. **HandsignLanguageRecognition.ipynd ("Battlefield")**:
  *  This is the main file for data collection, wrangling, model selection and evaluation.
2. **test.py**:
  *   Convenient way to test created model in livestream recognition. All variables are hard-coded, so any changes in the Battlefield file has to be transferred manually.
3. **saved_models folder**:
  *   All saved model in keras extension

__NOTE: Data was collected manually (as it can be seen in the Battlefield file) and will be replaced by more comprehensive datasets in the future.__
