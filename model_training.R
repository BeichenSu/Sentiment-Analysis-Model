# Sentiment Analysis Model
# Author: Beichen Su, Eric @ColourData
# Object: Input data -> Data clean up -> Tokenization -> Vectorization(By BOW)
# -> Train a neural network model

# Set working directory and seed
setwd("~/Sentiment_Model")
set.seed(13)

# Load required libraries
library(DBI)
library(RSQLServer)
library(readr)
library(jiebaR)
library(text2vec)
library(h2o)
library(tm)
library(sparsio)

# Call pre-defined funtions
source("functions.R")

# Define the customized parameter
num_of_records <- "50000"
min_word_count <- 5

# Load the training data(For some reason, the df is not constant, 
# save the working space everytime you have a new model, 
# in order to maintain the vocabulary)
df <- load_N_clean(num_of_records)

# Build the DTM matrix
dtm <- get_dtm(df, min_word_count)[[1]]

# write dtm out to read faster in h2o
write_svmlight(x = dtm, file = "dtm.txt", zero_based = FALSE)

# Initialize h2o
h2o.init(nthreads = -1)

# Read the data into h2o
hf <- h2o.importFile("dtm.txt")

# Seperate the training and test group, and making label factor
nrow = dim(hf)[1]
ncol = dim(hf)[2]
hf[,ncol] = h2o.asfactor(hf[,ncol])
sp = h2o.splitFrame(hf,ratios = 0.8)
hf_train = sp[[1]]
hf_test = sp[[2]]

# Now train the model, for detailed documentation, ?h2o.deeplearning
time_NN4x50 = system.time( 
  (md_NN4x50 = h2o.deeplearning(1:(ncol - 1), ncol, training_frame = hf_train, 
                                nfolds = 4, stopping_metric = "AUTO", model_id = "NN_4x50",
                                stopping_rounds = 3, stopping_tolerance = 1e-3, 
                                input_dropout_ratio = 0.2, hidden_dropout_ratios = c(0.3,0.3,0.3,0.3),
                                activation = "RectifierWithDropout",
                                hidden = c(50,50,50,50), reproducible = TRUE, seed = 123))
)

# Save model
h2o.saveModel(md_NN4x50, path = "NN4x50")

# Save the working space, you will need vocabulary to create DTM for the new incoming data
save.image("temp.RData")










