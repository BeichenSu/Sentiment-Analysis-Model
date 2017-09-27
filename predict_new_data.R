# Sentiment Analysis Model
# Author: Beichen Su, Eric @ColourData
# Object: Input data -> Data clean up -> Tokenization -> Vectorization(By BOW)
# -> Train a neural network model

# Set working directory and seed
setwd("~/Sentiment_Model")
set.seed(13)

# Call pre-defined funtions
source("functions.R")

# Load required libraries
library(DBI)
library(RSQLServer)
library(readr)
library(jiebaR)
library(text2vec)
library(h2o)
library(tm)
library(sparsio)

# Initialize h2o
h2o.init(nthreads = -1)

# Load the previous working space
# Need same vocabulary to create dtm for new data
load("First_work_space.RData")

# Get new data
daily_df <- get_dailyDF()

# Create vectorizer for dtm
vectorizer <- vocab_vectorizer(vocab)

# Create the DTM for the new data
new_dtm <- newDT_dtm(daily_df,vectorizer)

# Write it out for faster sparse matrix reading into h2o
write_svmlight(x = new_dtm, file = "new_dtm.txt", zero_based = FALSE)

# Bring the new dtm as h2o frame
new_hf <- h2o.importFile("new_dtm.txt")

# Load the saved model
my_model = h2o.loadModel("NN4x50/NN_4x50")

# Predictions
pred <- h2o.predict(my_model,new_hf)
pred <- as.data.frame(pred[,1])

# Output the prediction and combine with the sentence
daily_df <- cbind(daily_df,pred)
















