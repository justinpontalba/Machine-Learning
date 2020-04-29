# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:22:59 2020

@author: Justi

Advisement:
Please note that this code was derived from https://towardsdatascience.com/gentle-start-to-natural-language-processing-using-python-6e46c07addf3
The implementation of this code is for learning purposes only.
All rights are accredited to the original author.
"""

## Here we will learn how to identify what the web page is about using NLTK in Python

# grab a webpage and analyze the text to see what the page is about

# %%
import nltk
# nltk.download()

# %%
import urllib.request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


# %% crawl the webpage
websites = ["https://www.cbc.ca/news/canada/toronto/contracting-covid19-ontario-1.5548087",
            "https://www.theglobeandmail.com/canada/article-manitoba-to-ease-some-covid-19-restrictions-starting-monday/",
            "https://www.ctvnews.ca/health/coronavirus/"]
all_tokens = []
for site in websites:

    response = urllib.request.urlopen(site)
    html = response.read()
    # print(html)

    #  pulling data out of HTML and XML files using Beautiful Soup
    soup = BeautifulSoup(html,"lxml")
    text = soup.get_text(strip = True)
    # print(text)

    # tokenize text
    tokens = [t for t in text.split()]
    # print(tokens)

    # Count Word Frequency
    clean_tokens = []
    bad_chars = [';', ':', '!', "*","!","@", "#", "$", "%", "^","+","/", "|", "&" ] 
    for item in tokens[:]:
        if item in bad_chars:
            continue
        else:
            clean_tokens.append(item)


    sr= stopwords.words('english')
    # clean_tokens = tokens[:]
    for token in tokens:
        if token in stopwords.words('english'):
            clean_tokens.remove(token)
    
    all_tokens.extend(clean_tokens)

# %%
freq = nltk.FreqDist(all_tokens)
for key,val in freq.items():
    print(str(key) + ':' + str(val))
freq.plot(20, cumulative=False)