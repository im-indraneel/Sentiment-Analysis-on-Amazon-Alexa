#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis on Amazon Alexa

# In[44]:


import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

amazon_alexa_df = pd.read_csv("amazon_alexa.tsv", sep='\t')


# In[45]:


amazon_alexa_df.head()


# In[46]:


amazon_alexa_df.columns


# In[47]:


amazon_alexa_df = amazon_alexa_df[['rating','verified_reviews']]
print(amazon_alexa_df.shape)
amazon_alexa_df.head(20)


# In[48]:


amazon_alexa_df = amazon_alexa_df[amazon_alexa_df['rating'] != '4 and 5']
print(amazon_alexa_df.shape)
amazon_alexa_df.head(20)


# In[49]:


amazon_alexa_df["verified_reviews"].value_counts()


# In[50]:


sentiment_label = amazon_alexa_df.rating.factorize()
sentiment_label


# In[56]:


amazon_alexa = amazon_alexa_df.verified_reviews.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(amazon_alexa)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(amazon_alexa)
padded_sequence = pad_sequences(encoded_docs, maxlen=20)


# In[57]:


print(tokenizer.word_index)


# In[58]:


print(amazon_alexa[0])
print(encoded_docs[0])


# In[59]:


print(padded_sequence[0])


# In[60]:


embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model.summary()) 


# In[61]:


history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)


# In[62]:


plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")


# In[63]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plot.jpg")


# In[64]:


def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])


# In[65]:


test_sentence1 = "Love my Echo!."
predict_sentiment(test_sentence1)

test_sentence2 = "Without having a cellphone, I cannot use many!"
predict_sentiment(test_sentence2)


# In[ ]:




