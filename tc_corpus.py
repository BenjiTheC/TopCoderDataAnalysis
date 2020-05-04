""" The class taking in the data and tranform it into learning ready corpus."""

import os
import re
import json
import datetime
import difflib
import string
from collections import defaultdict

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, NavigableString, Tag

from sklearn.feature_extraction.text import CountVectorizer
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOP_WORDS

"""
Corpus container should have following functionality
- store the requirements by project id and challenge id and section name
- get the corpus by project / challenge
- get the similarity score
- get the tokenized & processed corpus (DTM) by proj id / cha id
- get the basic describe data of the number of word of a corput - by section or by challenge or by project

at least 2 DataFrame:
1. (project_id, challenge_id, section_name) MultiIndex DataFrame - lowercased stopwords removed punc removed url extracted text
2. (project_id, challenge_id) MultiIndex DataFrame - Bag of words -> Be careful about 10 fold verification!

"""

# Gensim stop word list is larger than scikit-learn & nltk stop wrods, but contains word "computer"
TC_STOP_WORDS = GENSIM_STOP_WORDS - {'computer'}

def remove_url(s):
    """ Remove url from given string s."""
    url_regex = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
    return re.sub(url_regex, '', s)

def remove_punctuation(s: str):
    """ Remove punctuation from given string s"""
    return s.translate(s.maketrans({p: None for p in string.punctuation}))

def remove_digits(s: str):
    """ Remove decimal digits or words containing decimal digits from given string s"""
    return re.sub(r'\w*\d\w*', '', s)

def remove_stop_words_from_str(s: str, stop_words=TC_STOP_WORDS, delimiter=' '):
    """ Remove the stop words using stop word list in Gensim"""
    return delimiter.join([word for word in s.split() if word.lower() not in stop_words])

def tokenize_str(s, min_len=2, max_len=20):
    """ Tokenize a string and return a list of str delimited by whitespace.
        Remove words too short (less than 2) or to long (greater than 20)
    """
    return [word for word in s.split() if min_len < len(word) < max_len]


class TopCoderCorpus:
    """ CorpusContainer takes in the data in json format and perform preprocessing
        operation for NLP experiement.

        > Note: it's currently designed only run fully in the memory
    """

    data_path = os.path.join(os.curdir, 'data')
    section_freq_threshold = 0.5

    def __init__(self, file_name):
        self.file_name = file_name
        self.titles, self.sectioned_requirements = self.process_detailed_req()
        # self.sections_similarity = self.calculate_section_similarity()

    def process_detailed_req(self) -> (pd.DataFrame, pd.DataFrame):
        """ Process the detailed requirements from loaded json
            Included preprocess methods:
            - lowercased
            - clean up white spaces
            - extract url
            - devide requirement paragraph by HTML header tag
        """
        processed_reqs = defaultdict(dict)
        processed_ttls = defaultdict(dict)

        with open(os.path.join(self.data_path, self.file_name)) as fjson:
            detailed_reqs = json.load(fjson)

        for req in detailed_reqs:
            processed_reqs[req['project_id']][req['challenge_id']] = self.sectionlize_requirements(req['requirements'])
            processed_ttls[req['project_id']][req['challenge_id']] = ' '.join(remove_digits(remove_punctuation(req['title'])).lower().split())

        flatten_reqs = {
                (project_id, challenge_id, sec_name): {'requirements_by_section': sec_text}
                for project_id, challenges in processed_reqs.items()
                for challenge_id, req in challenges.items()
                for sec_name, sec_text in req.items()
        } # multi-for dict comprehension > nested for loops with extra variable declared ;-)

        flatten_ttls = {(project_id, challenge_id): {'title': title} for project_id, challenges in processed_ttls.items() for challenge_id, title in challenges.items()}

        df_requirements = pd.DataFrame.from_dict(flatten_reqs, orient='index')
        df_requirements.index.names = ['project_id', 'challenge_id', 'section_name']

        df_titles = pd.DataFrame.from_dict(flatten_ttls, orient='index')
        df_titles.index.names = ['project_id', 'challenge_id']

        return df_titles, df_requirements

    def extract_txt_from_node(self, node, is_nav=False, delimiter=' '):
        """ Extract text from given node of a HTML parse tree, remove url, words with digits, punctuation."""
        text = node.strip() if is_nav else node.get_text()

        return delimiter.join(remove_digits(remove_punctuation(remove_url(text))).lower().split())

    def sectionlize_requirements(self, req):
        """ Aggregate the requirement paragraph by header tag. """
        sectioned_req_dct = defaultdict(list)
        soup = BeautifulSoup(req, 'html.parser')
        
        # There are some img tags and a tags that won't be extracted below, do it now.
        if soup.a:
            soup.a.decompose()
        if soup.img:
            soup.img.decompose()

        all_header_tags = soup.find_all(re.compile(r'^h'))
        
        if len(all_header_tags) == 0:
            return {'no_header_tag': self.extract_txt_from_node(soup)}
        
        for header in all_header_tags:
            section_name = self.extract_txt_from_node(header, delimiter='_')
            nxt_node = header
            while True:
                nxt_node = nxt_node.nextSibling
                
                if nxt_node is None:
                    break
                    
                if isinstance(nxt_node, NavigableString):
                    sectioned_req_dct[section_name].append(self.extract_txt_from_node(nxt_node, is_nav=True))
                if isinstance(nxt_node, Tag):
                    if nxt_node.name.startswith('h'):
                        break
                    sectioned_req_dct[section_name].append(self.extract_txt_from_node(nxt_node))
        
        return {sec_name: ' '.join(' '.join(sec_reqs).split()) for sec_name, sec_reqs in sectioned_req_dct.items()}

    def get_similarity_score(self, lst_of_str):
        """ Calculate the simliarity scroe from a list of strings"""
        seq_matcher = difflib.SequenceMatcher()
        similarity_score_sum = 0
        
        for idx, s in enumerate(lst_of_str[:-1]):
            seq_matcher.set_seq2(s)
            for s1 in lst_of_str[idx + 1:]:
                seq_matcher.set_seq1(s1)
                similarity_score_sum += round(seq_matcher.ratio(), 3)
                
        return round(similarity_score_sum / ((len(lst_of_str) * (len(lst_of_str) - 1)) / 2), 3)

    def calculate_section_similarity(self, threshold=section_freq_threshold):
        """ Find "common sections" of a project and calculate the similarity of text in the section of challenges"""
        sections_by_proj = defaultdict(lambda: defaultdict(dict))

        for project_id, challenges in self.sectioned_requirements.groupby(level=0):
            sec_name_freq = defaultdict(int)
            sec_text_by_name = defaultdict(list)
            num_of_challenges = len(challenges.index.unique(level='challenge_id'))

            for sec_name, sec_text in challenges.droplevel(level=[0, 1]).itertuples():
                if sec_name != 'no_header_tag':
                    sec_name_freq[sec_name] += 1
                    sec_text_by_name[sec_name].append(sec_text)

            for sec_name, freq in sec_name_freq.items():
                if freq / num_of_challenges > threshold:
                    sections_by_proj[project_id][sec_name] = {
                        'section_name_freq': round(freq / num_of_challenges, 2),
                        'lst_of_sec_text': sec_text_by_name[sec_name]
                    }

        section_similarity_score = defaultdict(lambda: defaultdict(dict))
        for project_id, req_sec in sections_by_proj.items():
            for sec_name, section in req_sec.items():
                section_similarity_score[project_id][sec_name] = {
                    'score': self.get_similarity_score(section['lst_of_sec_text']),
                    'freq': section['section_name_freq']
                }

        df_similarity_score = pd.DataFrame.from_dict(
            {(project_id, sec_name): sec for project_id, sections in section_similarity_score.items() for sec_name, sec in sections.items()},
            orient='index'
        )
        df_similarity_score.index.names = ['project_id', 'section_name']
        
        return df_similarity_score

    def get_challenge_req_sentences(self, as_dataframe=False):
        """ Process the sectioned requirements
            - concat all sections' text
            - remove stop words
            - tokenize words
        """
        challenge_req = self.sectioned_requirements.groupby(level=1).aggregate(' '.join).apply({'requirements_by_section': remove_stop_words_from_str})
        challenge_req.columns = ['requirements']
        return [tokenize_str(req) for cha_id, req in challenge_req.itertuples()] if not as_dataframe else challenge_req

if __name__ == '__main__':
    tccorpus = TopCoderCorpus('detail_requirements.json')
