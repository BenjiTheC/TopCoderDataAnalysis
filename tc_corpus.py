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
IMPORTANT:
The detailed requirement text data are from challenges which have valid project ids (!= -1)
and where the project has AT LEAST 10 challenges.

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
    file_name = 'detail_requirements.json'

    def __init__(self):
        self.titles, self.sectioned_requirements = self.process_detailed_req()

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

    def get_challenge_req(self):
        """ Process the sectioned requirements
            - concat all sections' text
            - remove stop words
            - tokenize words
        """
        challenge_req = self.sectioned_requirements.groupby(level=1).aggregate(' '.join)
        challenge_req.columns = ['requirements']
        return challenge_req

    # def get_challenge_req_sentences_no_overlap(self, select_overlap=False):
    #     """ Process the sectioned requirements
    #         - concat the distinct section
    #         - remove stop words
    #         - tokenize words

    #         By distinct that means the section's content are lower thant the similarity threshold
    #     """
    #     dropping_index = [] # collect the excluded index
    #     selective_dropping_content = []
    #     for (project_id, section_name), similarity_score, section_freqency in self.sections_similarity.loc[self.sections_similarity.score > self.section_similarity_threshold].itertuples():
    #         overlap_sections = self.sectioned_requirements.loc[pd.IndexSlice[project_id, :, section_name]]
            
    #         selective_dropping_content.append(overlap_sections.iloc[overlap_sections.requirements_by_section.str.len().argmax(), 0])
    #         dropping_index.extend(overlap_sections.index)

    #     challenge_req = self.sectioned_requirements.loc[~self.sectioned_requirements.index.isin(dropping_index)].groupby(level=1).aggregate(' '.join).apply({'requirements_by_section': remove_stop_words_from_str})
 
    #     if select_overlap:
    #         return [*[tokenize_str(req) for cha_id, req in challenge_req.itertuples()], *selective_dropping_content]
    #     else:
    #         return [tokenize_str(req) for cha_id, req in challenge_req.itertuples()]

if __name__ == '__main__':
    tccorpus = TopCoderCorpus()
