# Image-caption-generation-using-CNN-LSTM-architecture
<h2>INTRODUCTION</h2>
<ul>
  <li>Caption generation is an interesting artificial intelligence
problem where a descriptive sentence is generated for a given
    image.It involves the dual techniques from computer vision to
understand the content of the image and a language model from
the field of natural language processing to turn the
    understanding of the image into words in the right order.</li>
  <li>Image
captioning has various applications such as recommendations
in editing applications, usage in virtual assistants, for image
indexing, for visually impaired persons, for social media, and
    several other natural language processing applications.</li>
<li>Recently, deep learning methods have achieved state-ofthe-art
results on examples of this problem. It has been demonstrated
that deep learning models are able to achieve optimum results
in the field of caption generation problems. Instead of requiring
complex data preparation or a pipeline of specifically designed
models, a single end-to-end model can be defined to predict a
  caption, given a photo. </li>
  <li>In order to evaluate our model, we
measure its performance on the Flickr8K dataset using the
BLEU standard metric. These results show that our proposed
model performs better than standard models regarding image
    captioning in performance evaluation.</li> 
<h2>DATASET AND EVALUATION METRICS</h2>
For task of image captioning there are several annotated images
dataset are available. Most common of them are Pascal VOC
dataset, Flickr 8K and MSCOCO Dataset. Flickr 8K Image
captioning dataset [9] is used in the proposed model. Flickr 8K
is a dataset consisting of 8,092 images from the Flickr.com
website. This dataset contains collection of day-to-day activity
with their related captions. First each object in image is labeled
and after that description is added based on objects in an image.
We split 8,000 images from this corpus into three disjoint sets.
The training data (DTrain) has 6000 images whereas the
development and test dataset consist of 1000 images each.
In order to evaluate the image-caption pairs, we need to
evaluate their ability to associate previously unseen images and
captions with each other. The evaluation of model that
generates natural language sentence can be done by the BLEU
(Bilingual Evaluation Understudy) Score. It describes how
natural sentence is compared to human generated sentence. It
is widely used to evaluate performance of Machine translation.
Sentences are compared based on modified n-gram precision
method for generating BLEU score.<br>
Our model to caption images are built on multimodal recurrent
and convolutional neural networks. A Convolutional Neural
Network is used to extract the features from an image which is
then along with the captions is fed into an Recurrent Neural
Network. The architecture of the image captioning model is
shown in figure 1.<br>
![ima](https://user-images.githubusercontent.com/66600114/132738892-426bc3c8-0e84-49ac-96c5-6fa786c8dbf2.PNG)

The model consists of 3 phases:<br>
<h3>A. Image Feature Extraction</h3>
The features of the images from the Flickr 8K dataset is
extracted using the VGG 16 model due to the performance of
the model in object identification. The VGG is a convolutional
neural network which consists of consists of 16 layer which has
a pattern of 2 convolution layers followed by 1 dropout layers
until the fully connected layer at the end. The dropout layers
are present to reduce overfitting the training dataset, as this
model configuration learns very fast. These are processed by a
Dense layer to produce a 4096 vector element representation of
the photo and passed on to the LSTM layer.
<h3>B. Sequence processor</h3>
The function of a sequence processor is for handling the text
input by acting as a word embedding layer. The embedded layer
consists of rules to extract the required features of the text and
consists of a mask to ignore padded values. The network is then
connected to a LSTM for the final phase of the image
captioning.
<h3>C. Decoder</h3>
The final phase of the model combines the input from the Image
extractor phase and the sequence processor phase using an
additional operation then fed to a 256 neuron layer and then to
a final output Dense layer that produces a softmax prediction
of the next word in the caption over the entire vocabulary which
was formed from the text data that was processed in the
sequence processor phase. The structure of the network to
understand the flow of images and text is shown in the Figure
2.
<h2>TRAINING PHASE</h3>
During training phase we provide pair of input image and its
appropriate captions to the image captioning model. The VGG
model is trained to identify all possible objects in an image.
While LSTM part of model is trained to predict every word in
the sentence after it has seen image as well as all previous
words. For each caption we add two additional symbols to
denote the starting and ending of the sequence. Whenever stop
word is encountered it stops generating sentence and it marks
end of string. Loss function for model is calculated as, where I
represents input image and S represents the generated caption.
N is length of generated sentence. pt and St represent
probability and predicted word at the time t respectively.
During the process of training we have tried to minimize this
loss function.
<br>
<h2>RESULTS AND COMPARISION</h2>
The image captioning model was implemented and we were
able to generate moderately comparable captions with
compared to human generated captions. The VGG net model
first assigns probabilities to all the objects that are possibly
present in the image, as shown in Figure 3. The model converts
the image into word vector. This word vector is provided as
input to LSTM cells which will then form sentence from this
word vector. The generated sentences are shown in Fig 4.
Generated sentence are black dog runs into the ocean next to a
rock, while actual human generated sentences are black dog
runs into the ocean next to a pile of seaweed., black dog runs
into the ocean, a black dog runs into the ball, a black dog runs
to a ball. This results in a BLEU score of 57 for this image. 
<h2>CONCLUSION</h2>
Implemented a deep learning
approach for the captioning of images. The sequential API of
Keras was used with Tensorflow as a backend to implement the
deep learning architecture to achieve a effective BLEU score of 0.683 for our model.
The Bilingual Evaluation Understudy
Score, or BLEU for short, is a metric for evaluating a generated
sentence to a reference sentence. A perfect match results in a
score of 1.0, whereas a perfect mismatch results in a score of
0.0. 
