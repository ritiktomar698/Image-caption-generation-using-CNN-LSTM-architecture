# Image-caption-generation-using-CNN-LSTM-architecture
Caption generation is an interesting artificial intelligence
problem where a descriptive sentence is generated for a given
image. It involves the dual techniques from computer vision to
understand the content of the image and a language model from
the field of natural language processing to turn the
understanding of the image into words in the right order. Image
captioning has various applications such as recommendations
in editing applications, usage in virtual assistants, for image
indexing, for visually impaired persons, for social media, and
several other natural language processing applications.
Recently, deep learning methods have achieved state-ofthe-art
results on examples of this problem. It has been demonstrated
that deep learning models are able to achieve optimum results
in the field of caption generation problems. Instead of requiring
complex data preparation or a pipeline of specifically designed
models, a single end-to-end model can be defined to predict a
caption, given a photo. In order to evaluate our model, we
measure its performance on the Flickr8K dataset using the
BLEU standard metric. These results show that our proposed
model performs better than standard models regarding image
captioning in performance evaluation. 
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

