# notebook https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/2nd-place-coleridge-inference-code.ipynb
# Data files: https://www.kaggle.com/datasets/leecming/robertalabelclassifierrawipcc/download?datasetVersionNumber=1

from itertools import chain
import json
import os
import re
import regex
from collections import defaultdict, Counter
from typing import Dict, List

import pandas as pd
import nltk
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fuzzywuzzy import fuzz

from model import Model


class Model2(Model):
    PRETRAINED_DIR = "model2/classifier"
    TOKENIZER_DIR = "model2/tokenizer"
    MIN_PROB = 0.9
    HIGH_FREQ = 50
    MATCHING_THRESHOLD = 90

    DOWNLOAD_ERROR_MESSAGE = "The model needs the kaggle data to initialize its paramters. Please see kaggle_data/README.txt."

    @staticmethod
    def clean_text(txt):
        return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()

    def __init__(self, use_gpu:bool):
        self.init_params(use_gpu)

    def init_params(self, use_gpu:bool):
        nltk.download('punkt')
        self.use_gpu = use_gpu

        if use_gpu:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(Model2.PRETRAINED_DIR).half().cuda()
        else:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(Model2.PRETRAINED_DIR)

        self.classifier.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(Model2.TOKENIZER_DIR)


    def preprocess(self, json_text: str) -> str:

        json_text = list(map(lambda x: ' '.join(x.values()), json_text))
        json_text = ' '.join(json_text)
        json_text = ' '.join(json_text.split())
        sentences = '\n'.join(sent_tokenize(json_text))

        abbr_def_pairs = extract_abbreviation_definition_pairs(doc_text=sentences)

        return abbr_def_pairs


    def predict(self, abbr_def_pairs) -> List[str]:


        text = list(abbr_def_pairs.values())


        label_preds = []

        with torch.no_grad():
            for batch_idx_start in range(0, len(text), 64):
                batch_idx_end = min(batch_idx_start + 64, len(text))
                current_batch = text[batch_idx_start:batch_idx_end]
                batch_features = self.tokenizer(
                    current_batch,
                    truncation=True,
                    max_length=64,
                    padding='max_length',
                    add_special_tokens=True,
                    return_tensors='pt'
                )
                if self.use_gpu:
                    batch_features = {k: v.cuda() for k,v in batch_features.items()}

                model_output = self.classifier(**batch_features, return_dict=True)
                if self.use_gpu:
                    batch_preds = torch.nn.Softmax(-1)(model_output['logits'])[:, 1].cpu().numpy()
                else:
                    batch_preds = torch.nn.Softmax(-1)(model_output['logits'])[:, 1].numpy()

                label_preds.append(batch_preds)

        candidate_prob_mapping = dict(zip(text, chain.from_iterable(label_preds)))


        long_id_mapping = defaultdict(set) # Mapping of candidate to document ID
        clean_raw_mapping = dict()  # Mapping of cleaned candidate string to raw candidate string
        long_short_mapping = dict()  # Mapping of LONG-NAME candidate to its acronym

        for short, long in abbr_def_pairs.items():
            if candidate_prob_mapping[long] > Model2.MIN_PROB: # Only accept labels that meet the minimum probabiity
                cleaned_form = Model2.clean_text(long)
                # long_id_mapping[cleaned_form].add(curr_id)
                clean_raw_mapping[cleaned_form] = long

                # Store only acronyms that are longer than 3 characters
                if len(short) > 3:
                    long_short_mapping[cleaned_form] = short

        # high_prob_freq_labels = [k for k, v in long_id_mapping.items() if len(v) > Model2.HIGH_FREQ]

        assert os.path.exists("kaggle_data/train.csv"), Model2.DOWNLOAD_ERROR_MESSAGE
        def_labels = set(pd.read_csv('kaggle_data/train.csv')['cleaned_label'].drop_duplicates())
        # high_prob_freq_labels = set(high_prob_freq_labels).union(def_labels)
        high_prob_freq_labels = def_labels


        # TODO: Continue here.
        id_to_pred_mapping = defaultdict(list)
        for curr_label in long_id_mapping:
            for curr_id in long_id_mapping[curr_label]:
                id_to_pred_mapping[curr_id].append(curr_label)

        for curr_id, curr_pred_list in id_to_pred_mapping.items():
            # Sort in following descending priority (a definite training label, doc frequency, length of string)
            curr_pred_list = sorted(curr_pred_list,
                                    key=lambda x:(x in def_labels,len(long_id_mapping[x]), 1./len(x)), reverse=True)
            sieved_pred_list = []
            for curr_pred in curr_pred_list:
                match_found = False
                for other_pred in sieved_pred_list:
                    # Check if a candidate is too similar to a definite training label prediction
                    if fuzz.token_set_ratio(curr_pred, other_pred) > Model2.MATCHING_THRESHOLD and curr_pred not in def_labels and other_pred in def_labels:
                        match_found = True
                        break

                if not match_found and (len(long_id_mapping[curr_pred]) > Model2.HIGH_FREQ or curr_pred in def_labels
                                        or re.search('([A-Z][a-z]+ )+(Study|Survey)$', clean_raw_mapping[curr_pred])
                                        or re.search('(Study|Survey) of', clean_raw_mapping[curr_pred])):
                    sieved_pred_list.append(curr_pred)

                    # Add acronym as prediction if present in raw document text
                    if curr_pred in long_short_mapping and re.search(r' {} '.format(long_short_mapping[curr_pred]),
                                                                    id_to_raw_text[curr_id]):
                        sieved_pred_list.append((Model2.clean_text(long_short_mapping[curr_pred])))

            id_to_pred_mapping[curr_id] = set(sieved_pred_list)

        print(id_to_pred_mapping)



# Schwartz-Hearst code =========================================================
class Candidate(str):
    def __init__(self, value):
        super().__init__()
        self.start = 0
        self.stop = 0

    def set_position(self, start, stop):
        self.start = start
        self.stop = stop


def yield_lines_from_file(file_path):
    with open(file_path, 'rb') as f:
        for line in f:
            try:
                line = line.decode('utf-8')
            except UnicodeDecodeError:
                line = line.decode('latin-1').encode('utf-8').decode('utf-8')
            line = line.strip()
            yield line


def yield_lines_from_doc(doc_text):
    for line in doc_text.split("\n"):
        yield line.strip()


def best_candidates(sentence):
    """
    :param sentence: line read from input file
    :return: a Candidate iterator
    """

    if '(' in sentence:
        # Check some things first
        if sentence.count('(') != sentence.count(')'):
            raise ValueError("Unbalanced parentheses: {}".format(sentence))

        if sentence.find('(') > sentence.find(')'):
            raise ValueError("First parentheses is right: {}".format(sentence))

        close_index = -1
        while 1:
            # Look for open parenthesis. Need leading whitespace to avoid matching mathematical and chemical formulae
            open_index = sentence.find(' (', close_index + 1)

            if open_index == -1: break

            # Advance beyond whitespace
            open_index += 1

            # Look for closing parentheses
            close_index = open_index + 1
            open_count = 1
            skip = False
            while open_count:
                try:
                    char = sentence[close_index]
                except IndexError:
                    # We found an opening bracket but no associated closing bracket
                    # Skip the opening bracket
                    skip = True
                    break
                if char == '(':
                    open_count += 1
                elif char in [')', ';', ':']:
                    open_count -= 1
                close_index += 1

            if skip:
                close_index = open_index + 1
                continue

            # Output if conditions are met
            start = open_index + 1
            stop = close_index - 1
            candidate = sentence[start:stop]

            # Take into account whitespace that should be removed
            start = start + len(candidate) - len(candidate.lstrip())
            stop = stop - len(candidate) + len(candidate.rstrip())
            candidate = sentence[start:stop]

            if conditions(candidate):
                new_candidate = Candidate(candidate)
                new_candidate.set_position(start, stop)
                yield new_candidate


def conditions(candidate):
    """
    Based on Schwartz&Hearst

    2 <= len(str) <= 10
    len(tokens) <= 2
    re.search(r'\p{L}', str)
    str[0].isalnum()

    and extra:
    if it matches (\p{L}\.?\s?){2,}
    it is a good candidate.

    :param candidate: candidate abbreviation
    :return: True if this is a good candidate
    """
    viable = True
    if regex.match(r'(\p{L}\.?\s?){2,}', candidate.lstrip()):
        viable = True
    if len(candidate) < 2 or len(candidate) > 10:
        viable = False
    if len(candidate.split()) > 2:
        viable = False
    if not regex.search(r'\p{L}', candidate):
        viable = False
    if not candidate[0].isalnum():
        viable = False

    return viable


def get_definition(candidate, sentence):
    """
    Takes a candidate and a sentence and returns the definition candidate.

    The definition candidate is the set of tokens (in front of the candidate)
    that starts with a token starting with the first character of the candidate

    :param candidate: candidate abbreviation
    :param sentence: current sentence (single line from input file)
    :return: candidate definition for this abbreviation
    """
    # Take the tokens in front of the candidate
    tokens = regex.split(r'[\s\-]+', sentence[:candidate.start - 2].lower())
    # the char that we are looking for
    key = candidate[0].lower()

    # Count the number of tokens that start with the same character as the candidate
    first_chars = [t[0] for t in filter(None, tokens)]

    definition_freq = first_chars.count(key)
    candidate_freq = candidate.lower().count(key)

    # Look for the list of tokens in front of candidate that
    # have a sufficient number of tokens starting with key
    if candidate_freq <= definition_freq:
        # we should at least have a good number of starts
        count = 0
        start = 0
        start_index = len(first_chars) - 1
        while count < candidate_freq:
            if abs(start) > len(first_chars):
                raise ValueError("candidate {} not found".format(candidate))
            start -= 1
            # Look up key in the definition
            try:
                start_index = first_chars.index(key, len(first_chars) + start)
            except ValueError:
                pass

            # Count the number of keys in definition
            count = first_chars[start_index:].count(key)

        # We found enough keys in the definition so return the definition as a definition candidate
        start = len(' '.join(tokens[:start_index]))
        stop = candidate.start - 1
        candidate = sentence[start:stop]

        # Remove whitespace
        start = start + len(candidate) - len(candidate.lstrip())
        stop = stop - len(candidate) + len(candidate.rstrip())
        candidate = sentence[start:stop]

        new_candidate = Candidate(candidate)
        new_candidate.set_position(start, stop)
        return new_candidate

    else:
        raise ValueError('There are less keys in the tokens in front of candidate than there are in the candidate')


def select_definition(definition, abbrev):
    """
    Takes a definition candidate and an abbreviation candidate
    and returns True if the chars in the abbreviation occur in the definition

    Based on
    A simple algorithm for identifying abbreviation definitions in biomedical texts, Schwartz & Hearst
    :param definition: candidate definition
    :param abbrev: candidate abbreviation
    :return:
    """

    if len(definition) < len(abbrev):
        raise ValueError('Abbreviation is longer than definition')

    if abbrev in definition.split():
        raise ValueError('Abbreviation is full word of definition')

    s_index = -1
    l_index = -1

    while 1:
        try:
            long_char = definition[l_index].lower()
        except IndexError:
            raise

        short_char = abbrev[s_index].lower()

        if not short_char.isalnum():
            s_index -= 1

        if s_index == -1 * len(abbrev):
            if short_char == long_char:
                if l_index == -1 * len(definition) or not definition[l_index - 1].isalnum():
                    break
                else:
                    l_index -= 1
            else:
                l_index -= 1
                if l_index == -1 * (len(definition) + 1):
                    raise ValueError("definition {} was not found in {}".format(abbrev, definition))

        else:
            if short_char == long_char:
                s_index -= 1
                l_index -= 1
            else:
                l_index -= 1

    new_candidate = Candidate(definition[l_index:len(definition)])
    new_candidate.set_position(definition.start, definition.stop)
    definition = new_candidate

    tokens = len(definition.split())
    length = len(abbrev)

    if tokens > min([length + 5, length * 2]):
        raise ValueError("did not meet min(|A|+5, |A|*2) constraint")

    # Do not return definitions that contain unbalanced parentheses
    if definition.count('(') != definition.count(')'):
        raise ValueError("Unbalanced parentheses not allowed in a definition")

    return definition


def extract_abbreviation_definition_pairs(file_path=None,
                                          doc_text=None,
                                          most_common_definition=False,
                                          first_definition=False):
    abbrev_map = dict()
    list_abbrev_map = defaultdict(list)
    counter_abbrev_map = dict()
    omit = 0
    written = 0
    if file_path:
        sentence_iterator = enumerate(yield_lines_from_file(file_path))
    elif doc_text:
        sentence_iterator = enumerate(yield_lines_from_doc(doc_text))
    else:
        return abbrev_map

    collect_definitions = False
    if most_common_definition or first_definition:
        collect_definitions = True

    for i, sentence in sentence_iterator:
        # Remove any quotes around potential candidate terms
        clean_sentence = regex.sub(r'([(])[\'"\p{Pi}]|[\'"\p{Pf}]([);:])', r'\1\2', sentence)
        try:
            for candidate in best_candidates(clean_sentence):
                try:
                    definition = get_definition(candidate, clean_sentence)
                except (ValueError, IndexError) as e:
                    print("{} Omitting candidate {}. Reason: {}".format(i, candidate, e.args[0]))
                    omit += 1
                else:
                    try:
                        definition = select_definition(definition, candidate)
                    except (ValueError, IndexError) as e:
                        print("{} Omitting definition {} for candidate {}. Reason: {}".format(i, definition, candidate, e.args[0]))
                        omit += 1
                    else:
                        # Either append the current definition to the list of previous definitions ...
                        if collect_definitions:
                            list_abbrev_map[candidate].append(definition)
                        else:
                            # Or update the abbreviations map with the current definition
                            abbrev_map[candidate] = definition
                        written += 1
        except (ValueError, IndexError) as e:
            print("{} Error processing sentence {}: {}".format(i, sentence, e.args[0]))
    # print("{} abbreviations detected and kept ({} omitted)".format(written, omit))

    # Return most common definition for each term
    if collect_definitions:
        if most_common_definition:
            # Return the most common definition for each term
            for k,v in list_abbrev_map.items():
                counter_abbrev_map[k] = Counter(v).most_common(1)[0][0]
        else:
            # Return the first definition for each term
            for k, v in list_abbrev_map.items():
                counter_abbrev_map[k] = v[0]
        return counter_abbrev_map

    # Or return the last encountered definition for each term
    return abbrev_map
# Schwartz-Hearst code =========================================================

if __name__=="__main__":
    with open("kaggle_data/test/2f392438-e215-4169-bebf-21ac4ff253e1.json") as f:
        json_text = json.load(f)

    model = Model2(False)
    output = model.predict(model.preprocess(json_text))