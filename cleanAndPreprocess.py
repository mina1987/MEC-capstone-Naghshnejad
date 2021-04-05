import os
from os.path import join, getsize
import re
import nltk.data



text_dir = 'data/multi-document/DUC 2003/text'
def rewriteDocsIntoSingleHTMLFreeTex():
    for root, dirs, files in os.walk(text_dir):
        if dirs != []:
            for dir in dirs:
                print(dir)
                curr_dir = text_dir + '/' + dir
                new_file = join(curr_dir, 'merge.txt')
                f = open(new_file,"w+")
                for sub_root, sub_dirs, sub_files in os.walk(curr_dir):
                    for file in sub_files:
                        if file != 'merge.txt':
                            print(file)
                            with open(join(curr_dir, file), 'r') as reading_file:
                                text = reading_file.read()
                                result = re.search('<TEXT>([\s\S]*)<\/TEXT>', text)
                                if result is None:
                                    print(text)
                                text = result.group(1)
                                TAG_RE = re.compile(r'<[^>]+>')
                                text = TAG_RE.sub('', text)
                                f.write("%s" % text)
                f.close()

def splitTextIntoSentences():
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for root, dirs, files in os.walk(text_dir):
        if dirs != []:
            for dir in dirs:
                print(dir)
                curr_dir = text_dir + '/' + dir
                summaryFile = join(curr_dir, 'merge.txt')
                fp = open(summaryFile,"r")
                data = fp.read()
                TAG_RE = re.compile(r'\n')
                data = TAG_RE.sub(' ', data)
                sentences = tokenizer.tokenize(data)
                fp.close()
                summaryFile2 = join(curr_dir, 'merge.txt')
                f = open(summaryFile2,"w+")
                for sentence in sentences:
                    f.write("%s\n" % sentence)
                f.close()

rewriteDocsIntoSingleHTMLFreeTex()
splitTextIntoSentences()
