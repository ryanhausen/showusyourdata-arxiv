# Model notebook https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/3rd%20Mikhail%20Arkhipov/3rd%20place%20coleridge.ipynb

# TODO: It looks like this model uses the training data to build up a corpus
#       of words for pattern matching. We need to build this once and save it.
#       The code should also be refactored for our use.
from collections import defaultdict
import os
import json
from pathlib import Path
import re
import sys
from itertools import chain
from functools import partial
from typing import Callable, Counter, Dict, Iterable, List
from urllib.request import AbstractBasicAuthHandler

from model import Model


class Model3(Model):
    MODEL_DIR = "model3"
    MODEL_PARAMS = os.path.join(MODEL_DIR, "params.txt")
    MODEL_DATA_DIR = "kaggle_data"
    MODEL_KEYWORDS = {
        "Study",
        "Survey",
        "Assessment",
        "Initiative",
        "Data",
        "Dataset",
        "Database",
    }
    MODEL_STOPWORDS_PAR = [
        " lab",
        "centre",
        "center",
        "consortium",
        "office",
        "agency",
        "administration",
        "clearinghouse",
        "corps",
        "organization",
        "organisation",
        "association",
        "university",
        "department",
        "institute",
        "foundation",
        "service",
        "bureau",
        "company",
        "test",
        "tool",
        "board",
        "scale",
        "framework",
        "committee",
        "system",
        "group",
        "rating",
        "manual",
        "division",
        "supplement",
        "variables",
        "documentation",
        "format",
    ]

    TOKENIZE_PAT = re.compile("[\w']+|[^\w ]")
    CAMEL_PAT = re.compile(r"(\b[A-Z]+[a-z]+[A-Z]\w+)")
    BR_PAT = re.compile("\s?\((.*)\)")
    PREPS = {"from", "for", "of", "the", "in", "with", "to", "on", "and"}

    DOWNLOAD_ERROR_MESSAGE = "The model needs the kaggle data to initialize its paramters. Please see kaggle_data/README.txt."

    def __init__(self,):
        self.init_params()

    @staticmethod
    def get_parenthesis(t, ds):
        # Get abbreviations in the brackets if there are any
        cur_abbrs = re.findall(re.escape(ds) + '\s?(\([^\)]+\)|\[[^\]]+\])', t)
        cur_abbrs = [abbr.strip('()[]').strip() for abbr in cur_abbrs]
        cur_abbrs = [re.split('[\(\[]', abbr)[0].strip() for abbr in cur_abbrs]
        cur_abbrs = [re.split('[;,]', abbr)[0].strip() for abbr in cur_abbrs]
        cur_abbrs = [a for a in cur_abbrs if not any(ch in a for ch in '[]()')]
        cur_abbrs = [a for a in cur_abbrs if re.findall('[A-Z][A-Z]', a)]
        cur_abbrs = [a for a in cur_abbrs if len(a) > 2]
        cur_abbrs = [a for a in cur_abbrs if not any(tok.islower() for tok in Model3.TOKENIZE_PAT.findall(a))]
        fabbrs = []
        for abbr in cur_abbrs:
            if not (sum(bool(re.findall('[A-Z][a-z]+', tok)) for tok in Model3.TOKENIZE_PAT.findall(abbr)) > 2):
                fabbrs.append(abbr)
        return fabbrs


    def predict(self, text: Dict[str, str]) -> List[str]:

        predictions = []

        for sec in text:
            section_text = sec["text"]
            current_preds = []
            for paragraph in section_text.split("\n"):
                for sent in re.split("[\.]", paragraph):
                    for ds in self.datasets:
                        if ds in sent:
                            predictions.append(ds)
                            predictions.extend(Model3.get_paranthesis(sent, ds))

        return predictions



    def init_params(self) -> None:

        if not os.path.exists(Model3.MODEL_PARAMS):
            self.__generate_params()

        self.params = None

    def __generate_params(self) -> None:

        texts, train_labels = self.__get_raw_data()
        ssai_par_datasets = Model3.__tokenized_extract(texts, Model3.MODEL_KEYWORDS)
        words = list(chain(*[Model3.__tokenize(ds) for ds in ssai_par_datasets]))

        mapfilters = [
            MapFilter_AndThe(),
            MapFilter_StopWords(Model3.MODEL_STOPWORDS_PAR),
            MapFilter_IntroSSAI(Model3.MODEL_KEYWORDS, Model3.TOKENIZE_PAT),
            MapFilter_IntroWords(),
            MapFilter_BRLessThanTwoWords(Model3.BR_PAT, Model3.TOKENIZE_PAT),
            MapFilter_PartialMatchDatasets(ssai_par_datasets),
            MapFilter_TrainCounts(
                texts,
                ssai_par_datasets,
                Model3.__get_index(texts, words),
                "data",
                2,
                0.1,
                Model3.TOKENIZE_PAT,
            ),
            MapFilter_BRPatSub(Model3.BR_PAT),
        ]

        for f in mapfilters:
            ssai_par_datasets = f(ssai_par_datasets)

        train_labels_set = set(chain(*train_labels))
        # This line is in the original notebook, but doesn't seem to do anything
        #train_datasets = [ds for ds in train_labels_set if sum(ch.islower() for ch in ds) > 0 ]
        train_datasets = [Model3.BR_PAT.sub('', ds).strip() for ds in train_labels_set]
        datasets = set(ssai_par_datasets) | set(train_datasets)

        self.datasets = datasets


    @staticmethod
    def __get_index(texts, words):
        # Returns a dictionary where words are keys and values are indices
        # of documents (sentences) in texts, in which the word present
        index = defaultdict(set)
        words = set(words)
        words = {
            w
            for w in words
            if w.lower() not in Model3.PREPS and re.sub("'", "", w).isalnum()
        }
        for n, text in enumerate(texts):
            tokens = Model3.__tokenize(text)
            for tok in tokens:
                if tok in words:
                    index[tok].add(n)
        return index

    @staticmethod
    def __tokenize(text):
        return Model3.TOKENIZE_PAT.findall(text)

    @staticmethod
    def __clean_text(text):
        return re.sub("[^A-Za-z0-9]+", " ", str(text).lower()).strip()

    @staticmethod
    def __tokenized_extract(texts, keywords):
        # Exracts all mentions of the form
        # Xxx Xxx Keyword Xxx (XXX)
        connection_words = {"of", "the", "with", "for", "in", "to", "on", "and", "up"}
        datasets = []
        for text in texts:
            try:
                # Skip texts without parenthesis orXxx Xxx Keyword Xxx (XXX) keywords
                if "(" not in text or all(not kw in text for kw in keywords):
                    continue

                toks = list(Model3.TOKENIZE_PAT.finditer(text))
                toksg = [tok.group() for tok in toks]

                found = False
                current_dss = set()
                for n in range(1, len(toks) - 2):
                    is_camel = bool(Model3.CAMEL_PAT.findall(toksg[n + 1]))
                    is_caps = toksg[n + 1].isupper()

                    if (
                        toksg[n] == "("
                        and (is_caps or is_camel)
                        and toksg[n + 2] == ")"
                    ):
                        end = toks[n + 2].span()[1]
                        n_capi = 0
                        has_kw = False
                        for tok, tokg in zip(toks[n - 1 :: -1], toksg[n - 1 :: -1]):
                            if tokg in keywords:
                                has_kw = True
                            if (
                                tokg[0].isupper()
                                and tokg.lower() not in connection_words
                            ):
                                n_capi += 1
                                start = tok.span()[0]
                            elif tokg in connection_words or tokg == "-":
                                continue
                            else:
                                break
                        if n_capi > 1 and has_kw:
                            ds = text[start:end]
                            datasets.append(ds)
                            found = True
                            current_dss.add(ds)
            except:
                print(text)

        return datasets


    def __get_raw_data(self):
        data_dirs = os.listdir(Model3.MODEL_DATA_DIR)
        assert "train" in data_dirs, Model3.DOWNLOAD_ERROR_MESSAGE
        assert "test" in data_dirs, Model3.DOWNLOAD_ERROR_MESSAGE
        assert "train.csv" in data_dirs, Model3.DOWNLOAD_ERROR_MESSAGE

        # we only need pandas for model param generation
        import pandas as pd

        sentencizer = DotSplitSentencizer(True)
        df = pd.read_csv(os.path.join(Model3.MODEL_DATA_DIR, "train.csv"))

        samples = {}
        for _, (
            idx,
            pub_title,
            dataset_title,
            dataset_label,
            cleaned_label,
        ) in df.iterrows():
            if idx not in samples:
                with open(os.path.join(Model3.MODEL_DATA_DIR, "train", (idx + ".json"))) as fp:
                    data = json.load(fp)
                samples[idx] = {
                    "texts": [sec["text"] for sec in data],
                    "dataset_titles": [],
                    "dataset_labels": [],
                    "cleaned_labels": [],
                    "pub_title": pub_title,
                    "idx": idx,
                }
            samples[idx]["dataset_titles"].append(dataset_title)
            samples[idx]["dataset_labels"].append(dataset_label)
            samples[idx]["cleaned_labels"].append(cleaned_label)

        train_ids = []
        train_texts = []
        train_labels = []
        for sample_dict in samples.values():
            train_ids.append(sample_dict["idx"])
            texts = sample_dict["texts"]
            if sentencizer is not None:
                texts = list(chain(*[sentencizer(text) for text in texts]))
            train_texts.append(texts)
            train_labels.append(sample_dict["dataset_labels"])

        test_texts = []
        test_ids = []
        for test_file in Path(os.path.join(Model3.MODEL_DATA_DIR,"test")).glob("*.json"):
            idx = test_file.name.split(".")[0]
            with open(test_file) as fp:
                data = json.load(fp)
            texts = [sec["text"] for sec in data]
            if sentencizer is not None:
                texts = list(chain(*[sentencizer(text) for text in texts]))

            test_texts.append(texts)
            test_ids.append(idx)

        return list(chain(*(train_texts + test_texts))), train_labels


class MapFilter:
    def __init__(
        self, map_f: Callable = lambda x: x, filter_f: Callable = lambda x: True
    ):
        self.map_f = map_f
        self.filter_f = filter_f

    def __call__(self, input: Iterable) -> Iterable:
        return map(self.map_f, filter(self.filter_f, input))


class MapFilter_AndThe(MapFilter):
    def __init__(self):
        pat = re.compile(" and [Tt]he ")
        map_f = lambda ds: pat.split(ds)[-1]
        super().__init__(map_f=map_f)


class MapFilter_StopWords(MapFilter):
    def __init__(self, stopwords, do_lower=True):
        lower_f = lambda x: x.lower() if do_lower else x
        stopwords = list(map(lower_f, stopwords))

        def filter_f(ds):
            ds_lower = lower_f(ds)
            return not any(sw in ds_lower for sw in stopwords)

        super().__init__(filter_f=filter_f)


class MapFilter_IntroSSAI(MapFilter):
    def __init__(self, keywords, tokenize_pattern):
        connection_words = {'of', 'the', 'with', 'for', 'in', 'to', 'on', 'and', 'up'}

        def map_f(ds):
            toks_spans = list(tokenize_pattern.finditer(ds))
            toks = [t.group() for t in toks_spans]
            start = 0
            if len(toks) > 3:
                if toks[1] == 'the':
                    start = toks_spans[2].span()[0]
                elif toks[0] not in keywords and  toks[1] in connection_words and len(toks) > 2 and toks[2] in connection_words:
                    start = toks_spans[3].span()[0]
                elif toks[0].endswith('ing') and toks[1] in connection_words:
                    if toks[2] not in connection_words:
                        start_tok = 2
                    else:
                        start_tok = 3
                    start = toks_spans[start_tok].span()[0]
                return ds[start:]
            else:
                return ds

        super().__init__(map_f=map_f)


class MapFilter_IntroWords(MapFilter):
    def __init__(self):
        miss_intro_pat = re.compile('^[A-Z][a-z\']+ (?:the|to the) ')
        map_f = lambda ds: miss_intro_pat.sub('', ds)

        super().__init__(map_f)

class MapFilter_BRLessThanTwoWords(MapFilter):
    def __init__(self, br_pat, tokenize_pat):

        filter_f = lambda ds: len(tokenize_pat.find_alll(br_pat.sub("", ds))) > 2

        super().__init__(filter_f=filter_f)

class MapFilter_PartialMatchDatasets(MapFilter):

    def __init__(self, dataset, br_pat):

        counter = Counter(dataset)
        abbrs_used = set()
        golden_ds_with_br = []

        for ds, _ in counter.most_common():
            abbr = br_pat.findall(ds)[0]

            if abbr not in abbrs_used:
                abbrs_used.add(abbr)
                golden_ds_with_br.append(ds)

        filter_f = lambda ds: not any((ds in ds_) and (ds != ds_) for ds_ in golden_ds_with_br)

        super().__init__(filter_f=filter_f)


class MapFilter_TrainCounts(MapFilter):

    def __init__(
        self,
        texts,
        datasets,
        index,
        kw,
        min_train_count,
        rel_freq_threshold,
        tokenize_pat,
    ):
        # Filter by relative frequency (no parenthesis)
        # (check the formula in the first cell)
        tr_counts, data_counts = MapFilter_TrainCounts.get_train_predictions_counts_data(
            texts,
            MapFilter_TrainCounts.extend_parentehis(
                set(datasets)
            ),
            index,
            kw,
            tokenize_pat,
        )
        stats = {}

        for ds, count in Counter(datasets).most_common():
            stats[ds] = [
                count,
                tr_counts[ds],
                tr_counts[re.sub('[\s]?\(.*\)', '', ds)],
                data_counts[ds],
                data_counts[re.sub('[\s]?\(.*\)', '', ds)]
            ]

        def filter_f(ds):
            count, tr_count, tr_count_no_br, dcount, dcount_nobr = stats[ds]
            return (tr_count_no_br > min_train_count) and (dcount_nobr / tr_count_no_br > rel_freq_threshold)

        super().__init__(filter_f=filter_f)





    @staticmethod
    def extend_paranthesis(datasets):
        # Return each instance of dataset from datasets +
        # the same instance without parenthesis (if there are some)
        pat = re.compile('\(.*\)')
        extended_datasets = []
        for ds in datasets:
            ds_no_parenth = pat.sub('', ds).strip()
            if ds != ds_no_parenth:
                extended_datasets.append(ds_no_parenth)
            extended_datasets.append(ds)
        return extended_datasets


    @staticmethod
    def get_train_predictions_counts_data(texts, datasets, index, kw, tokenize_pat):
        # Returns N_data and N_total counts dictionary
        # (check the formulas in the first cell)
        pred_count = Counter()
        data_count = Counter()
        if isinstance(kw, str):
            kw = [kw]

        for ds in datasets:
            first_tok, *toks = tokenize_pat.findall(ds)
            to_search = None
            for tok in [first_tok] + toks:
                if index.get(tok):
                    if to_search is None:
                        to_search = set(index[tok])
                    else:
                        to_search &= index[tok]
            for doc_idx in to_search:
                text = texts[doc_idx]
                if ds in text:
                    pred_count[ds] += 1
                    data_count[ds] += int(any(w in text.lower() for w in kw))
        return pred_count, data_count


class MapFilter_BRPatSub(MapFilter):
    def __init__(self, br_pat):

        map_f = lambda ds: br_pat.sub("", ds)

        super().__init__(map_f=map_f)


class Sentencizer:
    def __init__(self, sentencize_fun: Callable, split_by_newline: bool = True) -> None:
        self.sentencize = sentencize_fun
        self.split_by_newline = split_by_newline

    def __call__(self, text: str) -> List[str]:
        if self.split_by_newline:
            texts = text.split("\n")
        else:
            texts = [text]
        sents = []
        for text in texts:
            sents.extend(self.sentencize(text))
        return sents


class DotSplitSentencizer(Sentencizer):
    def __init__(self, split_by_newline: bool) -> None:
        def _sent_fun(text: str) -> List[str]:
            return [sent.strip() for sent in text.split(".") if sent]

        super().__init__(_sent_fun, split_by_newline)



if __name__=="__main__":
    # input_json_image = sys.argv[1]
    input_json_image = "2010.00311/2010.00311.json"
    with open(input_json_image, "r") as f:
        text = json.load(f)

    predictions = Model3().predict(text)

    print("Model 3 output:", predictions)


# Imports from notebook ========================================================
# import re
# from collections import defaultdict, Counter
# from pathlib import Path
# from functools import partial
# import json
# from itertools import chain, combinations
# from typing import Callable, List, Union, Optional, Set, Dict

# import pandas as pd
# from tqdm import tqdm
# # Imports from notebook ========================================================

# TOKENIZE_PAT = re.compile("[\w']+|[^\w ]")
# CAMEL_PAT = re.compile(r"(\b[A-Z]+[a-z]+[A-Z]\w+)")
# BR_PAT = re.compile("\s?\((.*)\)")
# PREPS = {'from', 'for', 'of', 'the', 'in', 'with', 'to', 'on', 'and'}


# DATA_DIR = Path('/kaggle/input/coleridgeinitiative-show-us-the-data/')
# TRAIN_MARKUP_FILE = DATA_DIR / 'train.csv'


# def tokenize(text):
#     return TOKENIZE_PAT.findall(text)


# def clean_text(txt):
#     return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


# def remove_substring_predictions(preds):
#     preds = set(preds)
#     to_filter = set()
#     for d1, d2 in combinations(preds, 2):
#         if d1 in d2:
#             to_filter.add(d1)
#         if d2 in d1:
#             to_filter.add(d2)
#     return list(preds - to_filter)


# def filter_stopwords(datasets, stopwords, do_lower=True):
#     # Remove all instances that contain any stopword as a substring
#     filtered_datasets = []
#     if do_lower:
#         stopwords = [sw.lower() for sw in stopwords]
#     for ds in datasets:
#         ds_to_analyze = ds.lower() if do_lower else ds
#         if any(sw in ds_to_analyze for sw in stopwords):
#             continue
#         filtered_datasets.append(ds)
#     return filtered_datasets


# def extend_parentehis(datasets):
#     # Return each instance of dataset from datasets +
#     # the same instance without parenthesis (if there are some)
#     pat = re.compile('\(.*\)')
#     extended_datasets = []
#     for ds in datasets:
#         ds_no_parenth = pat.sub('', ds).strip()
#         if ds != ds_no_parenth:
#             extended_datasets.append(ds_no_parenth)
#         extended_datasets.append(ds)
#     return extended_datasets


# def filler_intro_words(datasets):
#     miss_intro_pat = re.compile('^[A-Z][a-z\']+ (?:the|to the) ')
#     return [miss_intro_pat.sub('', ds) for ds in datasets]


# def filter_parial_match_datasets(datasets):
#     # Some matches are truncated due to parsing errors
#     # or other factors. To remove those, we look for
#     # the most common form of the dataset and remove
#     # mentions, that are substrings of this form.
#     # Obviously, some true mentions might be dropped
#     # at this stage
#     counter = Counter(datasets)

#     abbrs_used = set()
#     golden_ds_with_br = []

#     for ds, count in counter.most_common():
#         abbr = BR_PAT.findall(ds)[0]
#         no_br_ds = BR_PAT.sub('', ds)

#         if abbr not in abbrs_used:
#             abbrs_used.add(abbr)
#             golden_ds_with_br.append(ds)

#     filtered_datasets = []
#     for ds in datasets:
#         if not any((ds in ds_) and (ds != ds_) for ds_ in golden_ds_with_br):
#             filtered_datasets.append(ds)
#     return filtered_datasets


# def filter_br_less_than_two_words(datasets):
#     filtered_datasets = []
#     for ds in datasets:
#         no_br_ds = BR_PAT.sub('', ds)
#         if len(tokenize(no_br_ds)) > 2:
#             filtered_datasets.append(ds)
#     return filtered_datasets


# def filter_intro_ssai(datasets):
#     # Filtering introductory words marked as a part of the mention by mistake
#     connection_words = {'of', 'the', 'with', 'for', 'in', 'to', 'on', 'and', 'up'}
#     keywords = {'Program', 'Study', 'Survey', 'Assessment'}
#     filtered_datasets = []
#     for ds in datasets:
#         toks_spans = list(TOKENIZE_PAT.finditer(ds))
#         toks = [t.group() for t in toks_spans]
#         start = 0
#         if len(toks) > 3:
#             if toks[1] == 'the':
#                 start = toks_spans[2].span()[0]
#             elif toks[0] not in keywords and  toks[1] in connection_words and len(toks) > 2 and toks[2] in connection_words:
#                 start = toks_spans[3].span()[0]
#             elif toks[0].endswith('ing') and toks[1] in connection_words:
#                 if toks[2] not in connection_words:
#                     start_tok = 2
#                 else:
#                     start_tok = 3
#                 start = toks_spans[start_tok].span()[0]
#             filtered_datasets.append(ds[start:])
#         else:
#             filtered_datasets.append(ds)
#     return filtered_datasets


# def get_index(texts: List[str], words: List[str]) -> Dict[str, Set[int]]:
#     # Returns a dictionary where words are keys and values are indices
#     # of documents (sentences) in texts, in which the word present
#     index = defaultdict(set)
#     words = set(words)
#     words = {w for w in words if w.lower() not in PREPS and re.sub('\'', '', w).isalnum()}
#     for n, text in tqdm(enumerate(texts), total=len(texts)):
#         tokens = tokenize(text)
#         for tok in tokens:
#             if tok in words:
#                 index[tok].add(n)
#     return index


# def get_train_predictions_counts_data(datasets, index, kw):
#     # Returns N_data and N_total counts dictionary
#     # (check the formulas in the first cell)
#     pred_count = Counter()
#     data_count = Counter()
#     if isinstance(kw, str):
#         kw = [kw]

#     for ds in tqdm(datasets):
#         first_tok, *toks = tokenize(ds)
#         to_search = None
#         for tok in [first_tok] + toks:
#             if index.get(tok):
#                 if to_search is None:
#                     to_search = set(index[tok])
#                 else:
#                     to_search &= index[tok]
#         for doc_idx in to_search:
#             text = texts[doc_idx]
#             if ds in text:
#                 pred_count[ds] += 1
#                 data_count[ds] += int(any(w in text.lower() for w in kw))
#     return pred_count, data_count


# def filter_by_train_counts(datasets, index, kw, min_train_count, rel_freq_threshold):
#     # Filter by relative frequency (no parenthesis)
#     # (check the formula in the first cell)
#     tr_counts, data_counts = get_train_predictions_counts_data(extend_parentehis(set(datasets)), index, kw)
#     stats = []

#     for ds, count in Counter(datasets).most_common():
#         stats.append([ds, count, tr_counts[ds], tr_counts[re.sub('[\s]?\(.*\)', '', ds)],
#                       data_counts[ds], data_counts[re.sub('[\s]?\(.*\)', '', ds)]])

#     filtered_datasets = []
#     for ds, count, tr_count, tr_count_no_br, dcount, dcount_nobr in stats:
#         if (tr_count_no_br > min_train_count) and (dcount_nobr / tr_count_no_br > rel_freq_threshold):
#             filtered_datasets.append(ds)
#     return filtered_datasets


# def filter_and_the(datasets):
#     pat = re.compile(' and [Tt]he ')
#     return [pat.split(ds)[-1] for ds in datasets]

# # NOTEBOOK SECTION: Data Utils
# class Sentencizer:
#     def __init__(self,
#                  sentencize_fun: Callable,
#                  split_by_newline: bool = True) -> None:
#         self.sentencize = sentencize_fun
#         self.split_by_newline = split_by_newline

#     def __call__(self, text: str) -> List[str]:
#         if self.split_by_newline:
#             texts = text.split('\n')
#         else:
#             texts = [text]
#         sents = []
#         for text in texts:
#             sents.extend(self.sentencize(text))
#         return sents


# class DotSplitSentencizer(Sentencizer):
#     def __init__(self,
#                  split_by_newline: bool) -> None:
#         def _sent_fun(text: str) -> List[str]:
#             return [sent.strip() for sent in text.split('.') if sent]
#         super().__init__(_sent_fun, split_by_newline)


# def get_coleridge_data(data_path: Union[str, Path],
#                        sentencizer: Optional[Sentencizer] = None) -> None:
#     data_path = Path(data_path)

#     df = pd.read_csv(data_path / 'train.csv')

#     samples = {}
#     for _, (idx, pub_title, dataset_title, dataset_label, cleaned_label) in tqdm(df.iterrows()):
#         if idx not in samples:
#             with open(data_path / 'train' / (idx + '.json')) as fp:
#                 data = json.load(fp)
#             samples[idx] = {'texts': [sec['text'] for sec in data],
#                             'dataset_titles': [],
#                             'dataset_labels': [],
#                             'cleaned_labels': [],
#                             'pub_title': pub_title,
#                             'idx': idx
#                             }
#         samples[idx]['dataset_titles'].append(dataset_title)
#         samples[idx]['dataset_labels'].append(dataset_label)
#         samples[idx]['cleaned_labels'].append(cleaned_label)


#     train_ids = []
#     train_texts = []
#     train_labels = []
#     for sample_dict in samples.values():
#         train_ids.append(sample_dict['idx'])
#         texts = sample_dict['texts']
#         if sentencizer is not None:
#             texts = list(chain(*[sentencizer(text) for text in texts]))
#         train_texts.append(texts)
#         train_labels.append(sample_dict['dataset_labels'])

#     test_texts = []
#     test_ids = []
#     for test_file in (data_path / 'test').glob('*.json'):
#         idx = test_file.name.split('.')[0]
#         with open(test_file) as fp:
#             data = json.load(fp)
#         texts = [sec['text'] for sec in data]
#         if sentencizer is not None:
#             texts = list(chain(*[sentencizer(text) for text in texts]))

#         test_texts.append(texts)
#         test_ids.append(idx)

#     return train_texts, train_ids, train_labels, test_texts, test_ids


# train_texts, train_ids, train_labels, test_texts, test_ids = get_coleridge_data(DATA_DIR, DotSplitSentencizer(True))
# train_labels_set = set(chain(*train_labels))

# # all sentences from train and test as a single list
# texts = list(chain(*(train_texts + test_texts)))


# # NOTEBOOK SECTION: Pattern Extractor

# def tokenzed_extract(texts, keywords):
#     # Exracts all mentions of the form
#     # Xxx Xxx Keyword Xxx (XXX)

#     connection_words = {'of', 'the', 'with', 'for', 'in', 'to', 'on', 'and', 'up'}
#     datasets = []
#     for text in tqdm(texts):
#         try:
#             # Skip texts without parenthesis orXxx Xxx Keyword Xxx (XXX) keywords
#             if '(' not in text or all(not kw in text for kw in keywords):
#                 continue

#             toks = list(TOKENIZE_PAT.finditer(text))
#             toksg = [tok.group() for tok in toks]

#             found = False
#             current_dss = set()
#             for n in range(1, len(toks) - 2):
#                 is_camel = bool(CAMEL_PAT.findall(toksg[n + 1]))
#                 is_caps = toksg[n + 1].isupper()

#                 if toksg[n] == '(' and (is_caps or is_camel) and toksg[n + 2] == ')':
#                     end = toks[n + 2].span()[1]
#                     n_capi = 0
#                     has_kw = False
#                     for tok, tokg in zip(toks[n - 1:: -1], toksg[n - 1:: -1]):
#                         if tokg in keywords:
#                             has_kw = True
#                         if tokg[0].isupper() and tokg.lower() not in connection_words:
#                             n_capi += 1
#                             start = tok.span()[0]
#                         elif tokg in connection_words or tokg == '-':
#                             continue
#                         else:
#                             break
#                     if n_capi > 1 and has_kw:
#                         ds = text[start: end]
#                         datasets.append(ds)
#                         found = True
#                         current_dss.add(ds)
#         except:
#             print(text)

#     return datasets


# def get_parenthesis(t, ds):
#     # Get abbreviations in the brackets if there are any
#     cur_abbrs = re.findall(re.escape(ds) + '\s?(\([^\)]+\)|\[[^\]]+\])', t)
#     cur_abbrs = [abbr.strip('()[]').strip() for abbr in cur_abbrs]
#     cur_abbrs = [re.split('[\(\[]', abbr)[0].strip() for abbr in cur_abbrs]
#     cur_abbrs = [re.split('[;,]', abbr)[0].strip() for abbr in cur_abbrs]
#     cur_abbrs = [a for a in cur_abbrs if not any(ch in a for ch in '[]()')]
#     cur_abbrs = [a for a in cur_abbrs if re.findall('[A-Z][A-Z]', a)]
#     cur_abbrs = [a for a in cur_abbrs if len(a) > 2]
#     cur_abbrs = [a for a in cur_abbrs if not any(tok.islower() for tok in tokenize(a))]
#     fabbrs = []
#     for abbr in cur_abbrs:
#         if not (sum(bool(re.findall('[A-Z][a-z]+', tok)) for tok in tokenize(abbr)) > 2):
#             fabbrs.append(abbr)
#     return fabbrs


# # NOTEBOOK SECTION: Evaluation

# def get_datasets():
#     STOPWORDS_PAR = [' lab', 'centre', 'center', 'consortium', 'office', 'agency', 'administration', 'clearinghouse',
#                      'corps', 'organization', 'organisation', 'association', 'university', 'department',
#                      'institute', 'foundation', 'service', 'bureau', 'company', 'test', 'tool', 'board', 'scale',
#                      'framework', 'committee', 'system', 'group', 'rating', 'manual', 'division', 'supplement',
#                      'variables', 'documentation', 'format']

#     filter_stopwords_par_data = partial(filter_stopwords, stopwords=STOPWORDS_PAR)

#     keywords = {'Study', 'Survey', 'Assessment', 'Initiative', 'Data', 'Dataset', 'Database'}

#     # Datasets
#     ssai_par_datasets = tokenzed_extract(texts, keywords)

#     words = list(chain(*[tokenize(ds) for ds in ssai_par_datasets]))
#     texts_index = get_index(texts, words)
#     filter_by_train_counts_filled = partial(filter_by_train_counts, index=texts_index,
#                                             kw='data', min_train_count=2, rel_freq_threshold=0.1)

#     filters = [filter_and_the, filter_stopwords_par_data, filter_intro_ssai, filler_intro_words,
#                filter_br_less_than_two_words, filter_parial_match_datasets, filter_by_train_counts_filled]

#     for filt in filters:
#         ssai_par_datasets = filt(ssai_par_datasets)

#     ssai_par_datasets = [BR_PAT.sub('', ds) for ds in ssai_par_datasets]

#     return ssai_par_datasets


# def solution():
#     predictions = defaultdict(set)
#     datasets = get_datasets()
#     train_datasets = [ds for ds in train_labels_set if sum(ch.islower() for ch in ds) > 0 ]
#     train_datasets = [BR_PAT.sub('', ds).strip() for ds in train_labels_set]
#     datasets = set(datasets) | set(train_datasets)
#     for filename in tqdm((DATA_DIR / 'test').glob('*')):
#         idx = filename.name.split('.')[0]
#         predictions[idx]
#         with open(filename) as fin:
#             data = json.load(fin)

#         for sec in data:
#             text = sec['text']
#             current_preds = []
#             for paragraph in text.split('\n'):
#                 for sent in re.split('[\.]', paragraph):
#                     for ds in datasets:
#                         if ds in sent:
#                             current_preds.append(ds)
#                             current_preds.extend(get_parenthesis(sent, ds))
#             predictions[idx].update(current_preds)
#         predictions[idx] = remove_substring_predictions(predictions[idx])

#     prediction_str_list = []
#     for idx, datasets in predictions.items():
#         datasets_str = '|'.join(clean_text(d) for d in sorted(set(datasets)))
#         prediction_str_list.append([idx, datasets_str])

#     with open('submission.csv', 'w') as fin:
#         for idx, datasets in [['Id', 'PredictionString']] + prediction_str_list:
#             fin.write(','.join([idx, datasets]) + '\n')


# solution()
