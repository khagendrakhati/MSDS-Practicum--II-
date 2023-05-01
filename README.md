# MSDS-Practicum II : Image Caption Generator using Deep Learning
# Project Background
Image Caption Generation is one of the application of Deep Learning. When we pass the image to the model and the model does some processing and generates captions as per its training. It is based on the computer vision and Natural Language Processing concepts.The project aims to use computers to analyze the context of an image and generate relevant captions. In this project, we implement the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short-term memory).
## CNN LSTM ARCHITECTURE
Convolutional Neural Network (CNN) is used for image feature extraction and Long Short-Term Memory Network (LSTM) is used for for Natural Language Processing (NLP).A CNN-LSTM architecture has wide-ranging applications as it stands at the helm of Computer Vision and Natural Language Processing. It allows us to use state of the art neural models for NLP tasks such as the transformer for sequential image and video data. At the same time, extremely powerful CNN networks can be used for sequential data such as the natural language.

## Data specifications
The project's goal is to predict the captions for the input image. For this project, we will be using the Flickr_8K dataset.The dataset contains 8k images with 5 captions per image. For input, the features are extracted from both the image and the text captions. The features will be concatenated to predict
the caption's next word. CNN is used for image recognition, while LSTM is used for text recognition. The BLEU Score is a metric
used to assess the performance of the trained model.

## Methodology

1. IMPORT MODULES
First, we have to import all the basic modules we will be needing for this project, such as

a) os - used to handle files using system commands.

b) pickle - used to store numpy features extracted

c) numpy - a Python library that can perform a wide range of mathematical operations on arrays.

d) tqdm - progress bar decorator for iterators. Includes a default range iterator printing to stderr.

e) VGG16, preprocess_input - imported modules for feature extraction from the image data

f) load_img, img_to_array - used for loading the image and converting the image to a numpy array

g) Tokenizer - used for loading the text as convert them into a token

h) pad_sequences - used for equal distribution of words in sentences filling the remaining spaces with zeros

i) plot_model - used to visualize the architecture of the model through different images

2.EXTRACT IMAGE FEATURES
For image Detecting, we are using a pre-trained model which is VGG16. VGG16 is already installed in the Keras library.

3.LOADING THE DATA FOR PREPROCESSING AND EXTRACTing THE IMAGE FEATURES

a. The dictionary 'features' is created and will be loaded with image data extracted features.

b. image path, target size=(224, 224)) - a custom dimension that will be used to resize the image when it is loaded into the
array

c. image.reshape((1, image.shape[0], image.shape[1], image.shape[2]) - reshaping image data for preprocessing in an RGB
image.

d. model.predict(image, verbose=0) - Image feature extraction

e. img name.split('.')[0] - Removed the image name from the extension in order to load only the image name.

CLEANING THE DATA
One of the main steps in NLP is to remove noise so that the machine can detect the patterns easily in the text. Noise will be present in the form of special characters such as hashtags, punctuation and numbers. All of which are difficult for computers to understand if they are present in the text. So we need to remove these for better results.

PREPROCESS TEXT DATA
Defined to clean and convert the text for quicker process and better results

MODEL CREATION

a. shape=(4096,) - output length of the features from the VGG model

b. Dense - single dimension linear layer array

c. Dropout() - used to add regularization to the data, avoiding over fitting & dropping out a fraction of the data from the layers

d. model.compile() - compilation of the model

e. loss=’sparse_categorical_crossentropy’ - loss function for category outputs

f. optimizer=’adam’ - automatically adjust the learning rate for the model over the no. of epochs

g. Model plot shows the concatenation of the inputs and outputs into a single layer
h. Feature extraction of image was already done using VGG, no CNN model was needed in this step.

![image](https://user-images.githubusercontent.com/90472656/235405824-683f90a1-d6e1-4ada-a72e-e4162bcaf7c8.png

TRAIN THE MODEL

a. steps = len(train) // batch_size - back propagation and fetch the next data

b. Loss decreases gradually over the iterations

c. Increase the no. of epochs for better results

d. Assign the no. of epochs and batch size accordingly for quicker results

![image](https://user-images.githubusercontent.com/90472656/235405937-0be6cdfb-fe5a-47be-a529-cfdee7e4037c.png)

VALIDATE THE DATA USING BLEU SCORE

 Bilingual Evaluation Understudy or BLEU score is used to evaluate the descriptions generated from translation and other Natural
Language Processing (NLP) applications. In a list of tokens, the BLEU Score is used to compare the predicted text to a reference
text. A BLEU Score greater than 0.4 is considered good; for a higher score, increase the number of epochs accordingly.

VISUALIZE THE RESULTS

Image caption generator formed. The actual captions of the image are printed first, followed by a predicted caption of the image.

![image](https://user-images.githubusercontent.com/90472656/235406177-89466cad-eebc-4ec0-a20e-8c0f4fc0fd31.png)

# References

https://www.math.ucla.edu/~minchen/doc/ImgCapGen.pdf

https://www.neuroquantology.com/data-cms/articles/20221021110155amNQ77261.pdf

https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8

https://www.analyticsvidhya.com/blog/2021/12/step-by-step-guide-to-build-image-caption-generator-using-deep-learning/

https://www.geeksforgeeks.org/image-caption-generator-using-deep-learning-on-flickr8k-dataset/






