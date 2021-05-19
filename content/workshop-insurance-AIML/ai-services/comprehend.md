---
title: "Lab: Comprehend"
chapter: false
weight: 22
---

## Introduction

Amazon Comprehend provides textual analysis for a given content. the current available insights are:

* Entities – extract a a list of entities, such as people, places, and locations, identified in a document.
* Key phrases – extracts key phrases that appear in a document.
* PII – analyzes and detect personal data that could be used to identify an individual, such as an address, bank account number, or phone number.
* Language – the dominant language in a document.
* Sentiment – the emotional sentiment of a document.
* Syntax – parses each word in your document and determines the part of speech for the word. For example, in the sentence "It is raining today in Seattle," "it" is identified as a pronoun, "raining" is identified as a verb, and "Seattle" is identified as a proper noun.
* Custom classifier - you can also build our own classifier by providing content and matching classification.
* Topic modeling - organize a large corpus of documents into topics or clusters that are similar based on the frequency of words within them.

Comprehend supports both synchronous and batch API calls so it can be used for large scale processing as well as realtime results.

## Sentiment and Language example

In this example, we will show how to integrate Amazon comprehend when building a model as part of a feature engineering.
Let's assume in our data collection we have a text field "description" but in its current shape it's not going to be useful for training a model.
Instead, we would want to process this field by calling comprehend to preform text analysis and extract the sentiment, and the language in the text thus creating two new features.

```python
import boto3

description = "River Place Leasing office is open and conducting tours. Contact us today to schedule an appointment and learn about our offerings: $1500 security deposit for qualified applicants! No pet fees!"

client = boto3.client('comprehend')

# detecting language from description field

response = client.detect_dominant_language(Text=description)
language = response['Languages'][0]['LanguageCode']
print(lagnuage)

# this should print 'en'

# detecting sentiment

response = client.detect_sentiment(
    Text=description,
    LanguageCode=language)
sentiment=response['Sentiment']
print(sentiment)

# this should print NEUTRAL
```

## Detecting Entities example
In addition to the previous example, we would also want to analyze the text and filter out only Organization with a high confidence entities.

```python
response = client.detect_entities(
    Text=description,
    LanguageCode=language
)
entities = []
for entity in response['Entities']:
   if entity['Score']>0.8 and entity['Type']=="ORGANIZATION":
      entities.append(entity['Text'])
print(entities)

# this should print ['River Place Leasing']
```

