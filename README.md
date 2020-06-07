# Kaggle-Google-Quest-QA-40th-Solution
Team: S.A.Y

Rank: 40th/1571, top 3%

Leaderboard: https://www.kaggle.com/c/google-quest-challenge/leaderboard

Below you could find a general walk through our solution with tricks for Kaggle competition - Google QUEST Q&A Labeling.

The purpose of this competition is to analyze StackExchange questions & answers and to predict whether the question is interesting, whether the answer is helpful or misleading etc.

# Data Exploration

Details from EDA could be found here: https://github.com/Sheng327/Kaggle-Google-Quest-QA/blob/master/Data%20Exploration.ipynb

1. After carefully examined natures of individual target variables, we decided to train two separate models. The first one was to fit Bert base models with only question title and question body. The second one was to fit with only question title and answer. In this way, almost all question bodies and answers can be fitted into models. 

2. Other features, like `url` and `category` should be take into consideration besides text features. 

3. Ratings are discrete in the training set, which means output post-processing could improve score.

# Feature Engineering

1. We extract URLs from the url column and implemented one-hot encoding to both the URLs and category columns. There are 64 new feature columns (51 for url and 13 for category) after one-hot encoding. 
New feature dimension is (n_sample, 64).

2. We introduced new features based on Universal Sentence Encoder. There are some target variables such as `answer_relevance` and `answer_satisfication` indicating that sentence similarity could contribute. The Universal Sentence Encoder (USE) encodes text into high dimensional vectors that can be used for diverse tasks. The input is the variable-length English text, and the output is a 512-dimensional vector. So, we encoded question title, question body, and answer into 512-dimensional vectors and then compared the L2 distance and cosine similarities between them.
New feature dimension for outputs from USE is (n_sample, 512*2) (because we feed into a pair of sentences into the model, dimension for each output is 512). New feature dimension for L2 distance and cosine similarities is (n_sample, 2) (one for L2 distance and one for cosine similarities).

To sum up, we have 512 + 512 + 64 + 2 = 1090 new features.

# Model Structure

We used a combo of two pretrained Roberta base models to train the dataset. The first Roberta handles the question title and question body pairs, while the second Roberta handles the question title and answer pairs. We only fitted texts into Roberta models at the moment and set aside the 1090 new features.

To prepare the inputs for RoBERTa model to meet the limits(maximum inputs), we trimmed head and tail part of the texts and believe in this way, the model would learn most from the texts.

The Roberta base model has 12 hidden layers and there are 13 layers in total (which includes the final output layer). Every layer has the output dimension batch_size x 512 x 768 (batch _size x maxseqlen x emb_size). We took the last three output layers out, concatenated them together and then applied an global average-pooling on it, because we think instead of the first unit of the output layer, the average over 512 tokens for each input in the batch from last three layers could perform better to capture the meaning of the whole sentence pair. Also, the model would learn some lower level features and added them to the final representation. 

Finally, we concatenated 1090 features with the average embedding and then added a fully connected layer with 21 units for the title-body pair model and 9 units for the title-answer pair model.

The final predictions are the average of 8 pairs of Roberta models (GroupKFold n_splits = 8).

# Customized Learning rate

A customized scheduler inherited from PolynomialDecay was used here to change the learning rate dynamically. We would like to see the learning rate go up from a low value and then gradually decrease. Because a large part of the model is from a pre-trained model and we do not want to touch that part much.

# Post-processing

Based on EDA we found that discretization of predictions for some challenging targets led to better spearman corr. We just assign the outputs with the values in training data for each column base on their L1 distance. 
