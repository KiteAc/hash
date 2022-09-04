# DeepHash and triplet learning with online triplet mining

PyTorch implementation of DeepHash and triplet networks for learning embeddings.

DeepHash is a model used to create Binary encodings of images for that can be used in image retrival systems.
Learning representations is one of the most important tasks in machine learning domain.The central idea of learning representations is to train a deep learning model to receive good feature embeddings essential for the tasks like image classification or retrieval. Assuming access to supervised data, triplet networks [1,2] are very popular approaches that can be applied to train informative representation on the output of the network. The general idea of the model is to take three images, anchor, positive (usually object from the same class as anchor) and negative (from different class), pass through the network that shares the parameters, and train the embedding on the last layer, which forces anchor and query images to be closer than anchor and negative ones, according to the assumed distance calculated on the embeddings (see fig. 1 for details). The goal of your work is to implement and verify the quality of triplet loss for challenging tasks like image retrieval and classification.

# Tasks

1. Read the papers that introduces triplet networks ([1] and [2]) carefully a couple of
times to be familiar with the content and implementation details.

2. Take one of the pretrained models (VGG16, VGG19, ResNet) and one of the two
benchmark datasets: Cifar10. Extract the features from the last layer
before classifcation) and evaluate the representation capabilties of the taken
model. For classifcation purposes, you can apply the K-Nearest Neigbour (KNN)
approach and to examine the retrieval capabilties, use mean average precision
(mAP) for evaluation. Use proper research methodology (test\train split, consistent
method for mAP evaluation), whichis going to be used in the following experiments.
Provide t-sne plots for latent representation for qualitative evaluation.

3. Implement triplet losses provided in [1] and [2] directly on the representation from
the model that you selected in point 2. Train the model with the triplet losses
provided in the papers and evaluate the quality of the embeddings following the
methodology assumed in point 2. The loss function in [2] contains a margin
parameter alpha, whichshould be selected according to the proper model selection
procedure (based on the separate validation set). Compare and discuss the results
obtained in point 2. with the results achieved for both triplet losses.

4. One of the main drawbacks of triplet networks is a large number of possible triplets
that can be constructed for training purposes. The improper selection of the
triplets can decrease the convergence speed drastically, or even result in providing
poor data representation. Implement 3-4 hard mining techniques (you can use those
mentioned in [2] and propose your own solution) and examine their quality in terms
of retrieval capabilities and speed of training.

5. For image retrieval tasks it is useful to
represent the data using compact binary codes. The representative binary codes
can be achieved by application of the triplet training framework. We encourage you
to propose the method of training compact binary codes on the output of the model
that you used in previous experiments. You can try to reimplement some ideas from
[3] or/and [4]. Do not forget to report the most interesting results using the
consistent research methodology ​ (this task is not mandatory).

# Installation

Requires [pytorch](http://pytorch.org/) 1.3 with torchvision 

# Notebooks

1. [Task 2](https://github.com/jjmachan/DeepHash/blob/master/nbs/task_2.ipynb) - In this task notebook we try to figure out the representation capabilites of the pretrained models. This gives us a baseline for the experiments that follow.

2. [Task 3](https://github.com/jjmachan/DeepHash/blob/master/nbs/Task%203.ipynb) - We train a Triplet loss functition to measure the improvement in representation capabilities of the models.

3. [Task 4 semiHardNegetive](https://github.com/jjmachan/DeepHash/blob/master/nbs/task4_semihardNegetive.ipynb) - Training for triplet loss is very hard to converge if the appropriate triplets are not selected so here we try to implement 2 different triplet selectors that are commonly used. This on is semihardNegetive seletion where we take the triplets that are not very hard. This leads to faster convergence.

4. [Task 4 randomNegetive](https://github.com/jjmachan/DeepHash/blob/master/nbs/task4_random_negetive.ipynb) - In this selector we try to find random triplets from the set.

5. [Task 5 alpha1->16](https://github.com/jjmachan/DeepHash/blob/master/nbs/task5_alpha1_16.ipynb) - This is the notebook that shows the most important research findings. In this we train the deepHash model to generate the HashCodes of the imput images. We also try out a new method of slowly increasing the hypermater that controls how well the loss function weights the quantisation values. It was slowly increased leading to much more efficient training.

6. [Task 5 alpha16](https://github.com/jjmachan/DeepHash/blob/master/nbs/task5_alpha16.ipynb) - In this notebook we trainin the DeepHash paper as it is given in [9].



# Code structure

- **datasets.py**
  - *SiameseMNIST* class - wrapper for a MNIST-like dataset, returning random positive and negative pairs
  - *TripletMNIST* class - wrapper for a MNIST-like dataset, returning random triplets (anchor, positive and negative)
  - *BalancedBatchSampler* class - BatchSampler for data loader, randomly chooses *n_classes* and *n_samples* from each class based on labels
  - *TripletCifar* cass - wrapper for the CIFAR dataset, returning random triplets (anchor, positive and negative)
  - *BalancedBatchSamplerCifar* cass - BatchSampler for dataloader, CIFAR doesn't work with the other sampler hence this one.
- **networks.py**
  - *EmbeddingNet* - base network for encoding images into embedding vector
  - *ClassificationNet* - wrapper for an embedding network, adds a fully connected layer and log softmax for classification
  - *SiameseNet* - wrapper for an embedding network, processes pairs of inputs
  - *TripletNet* - wrapper for an embedding network, processes triplets of inputs
- **losses.py**
  - *ContrastiveLoss* - contrastive loss for pairs of embeddings and pair target (same/different)
  - *TripletLoss* - triplet loss for triplets of embeddings
  - *OnlineContrastiveLoss* - contrastive loss for a mini-batch of embeddings. Uses a *PairSelector* object to find positive and negative pairs within a mini-batch using ground truth class labels and computes contrastive loss for these pairs
  - *OnlineTripletLoss* - triplet loss for a mini-batch of embeddings. Uses a *TripletSelector* object to find triplets within a mini-batch using ground truth class labels and computes triplet loss
- **trainer.py**
  - *fit* - unified function for training a network with different number of inputs and different types of loss functions
- **metrics.py**
  - Sample metrics that can be used with *fit* function from *trainer.py*
- **utils.py**
  - *PairSelector* - abstract class defining objects generating pairs based on embeddings and ground truth class labels. Can be used with *OnlineContrastiveLoss*.
    - *AllPositivePairSelector, HardNegativePairSelector* - PairSelector implementations
  - *TripletSelector* - abstract class defining objects generating triplets based on embeddings and ground truth class labels. Can be used with *OnlineTripletLoss*.
    - *AllTripletSelector*, *HardestNegativeTripletSelector*, *RandomNegativeTripletSelector*, *SemihardNegativeTripletSelector* - TripletSelector implementations



# References

[1] Hoffer, Elad, and Nir Ailon. ["Deep metric learning using triplet network."](https://arxiv.org/abs/1412.6622) ​ International
Workshop on Similarity-Based Pattern Recognition . ​ Springer, Cham, 2015.

[2] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. ["Facenet: A unified embedding
for face recognition and clustering."](https://arxiv.org/abs/1503.03832) Proceedings of the IEEE conference on computer vision
and pattern recognition. 2015.

[3] Zhuang, Bohan, et al. ["Fast training of triplet-based deep binary embedding networks."](https://arxiv.org/abs/1503.03832)
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

[4] Wang, Xiaofang, Yi Shi, and Kris M. Kitani. ["Deep supervised hashing with triplet labels."](https://arxiv.org/abs/1612.03900)
Asian conference on computer vision . ​ Springer, Cham, 2016.

[5] Raia Hadsell, Sumit Chopra, Yann LeCun, [Dimensionality reduction by learning an invariant mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), CVPR 2006

[6] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. [Facenet: A unified embedding for face recognition and clustering.](https://arxiv.org/abs/1503.03832) CVPR 2015

[7] Alexander Hermans, Lucas Beyer, Bastian Leibe, [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737), 2017

[8] Brandon Amos, Bartosz Ludwiczuk, Mahadev Satyanarayanan, [OpenFace: A general-purpose face recognition library with mobile applications](http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf), 2016

[9] Yi Sun, Xiaogang Wang, Xiaoou Tang, [Deep Learning Face Representation by Joint Identification-Verification](http://papers.nips.cc/paper/5416-deep-learning-face-representation-by-joint-identification-verification), NIPS 2014
