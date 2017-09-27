# Create a connection to SQL Server
# Making Query and Fetch it to a data frame
# Unique lines will be taken and NAs will be removed(There is Actually no NA)
# Regulate column names of the data frame: BreakDownContent, SentimentKey
# Will return a data frame with clean data for vectorization
# Input: rows to be taken from db, need to be a character/string
load_N_clean <- function(num_of_records) {
  con <- dbConnect(RSQLServer::SQLServer(), server = "TEST", database = 'SparkTest')
  query <- paste("SELECT TOP",as.character(num_of_records),"[BreakDownContent],[QuerySentimentType]  FROM [SparkTest].[dbo].[ahwom_content_label] WHERE [A] is not NULL and [B] is not NULL")
  res <- dbSendQuery(con, query)
  df <- dbFetch(res)
  df <- unique(df)
  dbClearResult(res)
  colnames(df) <- c('BreakDownContent', 'SentimentKey')
  df
}

# Now fit the daily breakdowncontent
get_dailyDF <- function() {
  query <- 'SELECT [BreakDownID],[BreakDownContent]
  FROM [SparkTest].[dbo].[AutohomePRCBreakDown_Spark]
  where convert(date,createtime)=convert(date,getdate())'
  con <- dbConnect(RSQLServer::SQLServer(), server = "TEST", database = 'SparkTest')
  res <- dbSendQuery(con,query)
  daily_df <- dbFetch(res)
  dbClearResult(res)
  daily_df
}

# Remove stop words function
# Input: sentence token, stop words dictionary
# Output: sentence token with stop words removed
removewords = function(target_words,stop_words){
  target_words <- target_words[target_words%in%stop_words==FALSE]
  return(target_words)
}

# Tokenize the sentences, remove the stopping words
# by customized user dictionary
# Input: Sentence with labels, BreakDownContent, SentimentKey 
# Output: List of lists of words for each sentence, no label
sentence_token <- function(sentence) {
  # Input all kinds of existing dictionaries
  A_words <- readLines("A_words.txt", encoding = 'UTF-8')
  B_words <- readLines("B_words.txt", encoding = 'UTF-8')
  stopping_words <- readLines("stopping_words.txt", encoding = 'UTF-8')
  
  # Merge dictionaries
  mydict <- c(A_words,B_words)
  
  # Pick the unique words
  mydict <- unique(mydict)
  
  # Remove numbers from the sentence
  sentence <- removeNumbers(sentence)
  # Initialize the segment engine with user dictionary
  engine <- worker()
  new_user_word(engine,mydict)
  
  sentence <- gsub("\\.","",sentence)
  
  # Sentence segment without stopping words
  segwords <- sapply(sentence, segment, engine)
  
  # Remove stopping words
  return(sapply(segwords, removewords, stopping_words))
}


# Vectorize the sentence by the previous defined tokenization funtion
# Build document-term matrix(DTM), and combine with Sentiment Key
# The vocabulary is built on 2-gram
# Input: Sentence with labels (BreakDownContent, SentimentKey), prune-vocab min count
# Output: DTM matrix and label as a whole matrix, labels on the last column(-1,0,1 for Neg,Norec,Pos)
# and the vocabulary
get_dtm <- function(df, count) {
  sentence <- df$BreakDownContent
  # Create the text2vec chinese token
  it <- itoken(sentence,
               tokenizer = sentence_token, 
               progressbar = FALSE)
  
  # Build a text2vec vocabulary with n gram
  vocab <- create_vocabulary(it, ngram = c(1L, 2L))
  
  # Adjust the minimum count here, reduce the dimension
  vocab <- vocab %>% prune_vocabulary(term_count_min = count, 
                                      doc_proportion_max = 0.5)
  # Build the vectorizer function based on the vocabulary
  vectorizer <- vocab_vectorizer(vocab)
  
  # Build document-term matrix(DTM), and combine with Sentiment Key
  dtm <- create_dtm(it, vectorizer)
  
  # Transform labels into number
  df$SentimentKey[which(df$SentimentKey == "NoRec")] <-  0
  df$SentimentKey[which(df$SentimentKey == "Positive")] <- 1
  df$SentimentKey[which(df$SentimentKey == "Negative")] <- -1
  df$SentimentKey <- as.numeric(df$SentimentKey)
  
  # Combine DTM matrix with labels
  dtm <- cbind(dtm,df$SentimentKey)
  list(dtm,vocab)
}



# Pre-process new data, using the same vectorizer to create a dtm matrix without label
# And bring it the the h2o frame
# Input: data frame of sentences(Marked as BreakDownContent)
# Output: a new DTM matrix
newDT_dtm <- function(newDT, vectorizer) {
  newDT <- newDT$BreakDownContent
  it <- itoken(newDT,
               tokenizer = sentence_token, 
               progressbar = FALSE)
  dtm <- create_dtm(it, vectorizer)
}
