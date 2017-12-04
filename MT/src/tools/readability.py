#!/usr/bin/python

from __future__ import division
import collections
import string

def getTextScores(text, locale='en_US', simplewordlist=[]):
    """
    Calculates several text scores based on a piece of text.
    A custom locale can be provided for the hyphenator, which
    maps to a Myspell hyphenator dictionary.  If the locale
    is a file descriptor or file path the dictionary at that
    path will be used instead of those in the default Myspell
    location.
    The simple word list should be provided in lower case.
    """
    #from nltk.tokenize import sent_tokenize
    from hyphenator import Hyphenator
    import re
    import os

    #check if the locale is supplied as a file
    if os.path.exists(locale):
        hyphenator = Hyphenator(locale)
    else:
        hyphenator = Hyphenator(os.path.dirname(os.path.realpath(__file__)) + "/dict/hyph_" + locale + ".dic")

    scores = {
        'sent_count': 0,              # nr of sentences
        'word_count': 0,              # nr of words
        'letter_count':0,             # nr of characters in words (no spaces)
        'syll_count': 0,              # nr of syllables
        'polysyllword_count': 0,      # nr of polysyllables (words with more than 3 syllables)
        'simpleword_count': 0,        # nr of simplewords (depends on provided list)
        'sentlen_average': 0,         # words per sentence
        'wordlen_average': 0,         # syllables per word
        'wordletter_average': 0,      # letters per word
        'wordsent_average': 0         # sentences per word
    }
    '''if isinstance(text,unicode):
        sentences = sent_tokenize(text.encode('utf8'))
    else:
        sentences = sent_tokenize(text)'''
    sentences = text.split("\n")
    scores['sent_count'] = len(sentences)
    
    for s in sentences:
        words = re.findall(r'\w+', unicode(s.decode('utf-8')), flags = re.UNICODE)
        #words = s.split()

        scores['word_count'] = scores['word_count'] + len(words)

        for w in words:
            if w in string.punctuation:
                syllables_count = 1 #changed it according to wei's paper
            else:
                syllables_count = hyphenator.inserted(w).count('-') + 1
            scores['letter_count'] = scores['letter_count'] + len(w)
            scores['syll_count'] = scores['syll_count'] + syllables_count
            
            if syllables_count > 2:
                scores['polysyllword_count'] = scores['polysyllword_count'] + 1

            if simplewordlist:
                if w.lower() in simplewordlist:
                    scores['simpleword_count'] = scores['simpleword_count'] + 1


    if scores['sent_count'] > 0:
        scores['sentlen_average'] = scores['word_count'] / scores['sent_count']

    if scores['word_count'] > 0:
        scores['wordlen_average'] = scores['syll_count'] / scores['word_count']
        scores['wordletter_average'] = scores['letter_count'] / scores['word_count']
        scores['wordsent_average'] = scores['sent_count'] / scores['word_count']
    return scores


def getMinimumAgeFromUsGrade(us_grade):
    """
    The age has a linear relation with the grade.
    http://en.wikipedia.org/wiki/Education_in_the_United_States#School_grades
    """
    return int(round(us_grade + 5))

def flesch(text):
    scores = getTextScores(text, 'en_US')
    reading_ease = 206.835 - ( 1.015 * scores['sentlen_average'] ) - ( 84.6 * scores['wordlen_average'] )
    return reading_ease


def FleschKincaid(text, unk_offset):
    text = text.replace("<s>", "")
    text = text.replace("</s>", "").strip()
    scores = getTextScores(text, 'en_US')
    original_score = (0.39 * scores['sentlen_average']) + (11.8 * scores['wordlen_average']) - 15.59
    if scores["word_count"] > 0:
        now_score = original_score + unk_offset * text.count("<unk>") / scores['word_count']
    else:
        now_score = original_score
    return now_score

def FleschKincaid_len(text):
    #text = text.encode("utf-8")
    text = text.replace("<s>", "")
    text = text.replace("</s>", "").strip()
    scores = getTextScores(text, 'en_US')
    return (0.39 * scores['sentlen_average']) + (11.8 * scores['wordlen_average']) - 15.59, scores["word_count"]

def get_FK_bin(text, bins, unk_offset):
    score = FleschKincaid(text, unk_offset)
    for i in range(len(bins)):
        if score < bins[i]:
            return i
    return len(bins)


def work(inp1, inp2):
    scores = []
    for inp in [inp1, inp2]:
        for st in inp:
            scores.append(FleschKincaid(st))
    scores = sorted(scores)
    l = len(scores)
    k = 3
    score_divide = []
    for i in range(1, k):
        score_divide.append(scores[int(l * i / k)])
    print score_divide

if __name__ == "__main__":
    #print FleschKincaid("<unk> <unk> <unk>")
    #print FleschKincaid("error message logged")
    #print FleschKincaid("The military said many of the banned jobs will start opening up this year .")
    #print FleschKincaid("The military said many of the banned jobs will start opening this year .")
    st = "We are close to wrapping up our 10 week Rails Course .\nThis week we will cover a handful of topics commonly encountered in Rails projects .\nWe then wrap up with part 2 of our Reddit on Rails exercise!\nBy now you should be hard at work on your personal projects.\nThe students in the course just presented in front of the class with some live demos and a brief intro to to the problems their app were solving.\nMaybe set aside some time this week to show someone your progress, block off 5 minutes and describe what goal you are working towards, the current state of the project (is it almost done, just getting started, needs UI, etc.), and then show them a quick demo of the app.\n Explain what type of feedback you are looking for (conceptual, design, usability, etc.) and see what they have to say .\nAs we are wrapping up the course you need to be focused on learning as much as you can , but also making sure you have the tools to succeed after the class is over ."
    print FleschKincaid(st, 100000)
    print flesch(st)
    #normal_file = open('../../../data/newsela/train.normal.tok', "r")
    #simple_file = open('../../../data/newsela/train.simple.tok', "r")
    #work(normal_file, simple_file)
    threshold = [5.9, 9.5]
    '''threshold = [3, 5, 7, 9, 11, 13]
    d = collections.defaultdict(int)
    for st in simple_file:
        #d[int(len(st.strip().split()) / 5)] += 1
        d[get_FK_bin(st, threshold)] += 1
    for st in normal_file:
        d[get_FK_bin(st, threshold)] += 1
    for k in range(10):
        print k, d[k]
    print get_FK_bin("his father-in-law is the leader of vietnam .", threshold)'''
    #It is actually dividing both normal and simple to five categories
    '''cnt = 0
    eq = 0
    for n, s in zip(normal_file, simple_file):
        a = get_FK_bin(n, threshold)
        b = get_FK_bin(s, threshold)
        cnt += 1
        if a == b:
            eq += 1
        if cnt > 1000:
            break
    print eq * 1.0 / cnt'''
