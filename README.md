# Extractive-Domain-Independent-Graph-Based-Multi-Document-Text-Summarization

This is my final project for CS 224U at Stanford University.

This repository includes:
* baselineMultiDocument.py => an implementation of the baseline model for multi-document summarization. The baseline is simply LexRank.
* baselineSingleDocument.py => an implementation of the baseline model for single-document summarization. The baseline is simply LexRank.
* cleanAndPreprocess.py => code I used to clean and preprocess the data. The data I used was taken from DUC 2003 and DUC 2004.
* keyPhraseExtractor.py => extracts key phrases from the given merged document (all of the documents merged) and includes functions to return the scores relating to those key phrases for the various approaches. Refer to paper to understand.
* myRouge.py => my implementation of the ROUGE evaluation metric.
* mySummarizerMultipleDocument.py => my model for the multi-document text summarizer. This file brings all of the components discussed above together.
* Paper.pdf => my final paper for the project with the full details and rationale for the project. 
