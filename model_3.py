# Model notebook https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/3rd%20Mikhail%20Arkhipov/3rd%20place%20coleridge.ipynb

from collections import defaultdict
import os
import json
from pathlib import Path
import re
import sys
from itertools import chain
from functools import partial
from typing import Callable, Counter, Dict, Iterable, List

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

    def __init__(
        self,
    ):
        self.init_params()

    @staticmethod
    def get_parenthesis(t, ds):
        # Get abbreviations in the brackets if there are any
        cur_abbrs = re.findall(re.escape(ds) + "\s?(\([^\)]+\)|\[[^\]]+\])", t)
        cur_abbrs = [abbr.strip("()[]").strip() for abbr in cur_abbrs]
        cur_abbrs = [re.split("[\(\[]", abbr)[0].strip() for abbr in cur_abbrs]
        cur_abbrs = [re.split("[;,]", abbr)[0].strip() for abbr in cur_abbrs]
        cur_abbrs = [a for a in cur_abbrs if not any(ch in a for ch in "[]()")]
        cur_abbrs = [a for a in cur_abbrs if re.findall("[A-Z][A-Z]", a)]
        cur_abbrs = [a for a in cur_abbrs if len(a) > 2]
        cur_abbrs = [
            a
            for a in cur_abbrs
            if not any(tok.islower() for tok in Model3.TOKENIZE_PAT.findall(a))
        ]
        fabbrs = []
        for abbr in cur_abbrs:
            if not (
                sum(
                    bool(re.findall("[A-Z][a-z]+", tok))
                    for tok in Model3.TOKENIZE_PAT.findall(abbr)
                )
                > 2
            ):
                fabbrs.append(abbr)
        return fabbrs

    def predict(self, text: Dict[str, str]) -> List[str]:

        predictions = []

        for sec in text:
            section_text = sec["text"]

            for paragraph in section_text.split("\n"):
                for sent in re.split("[\.]", paragraph):
                    for ds in self.datasets:
                        if ds in sent:
                            predictions.append(ds)
                            predictions.extend(Model3.get_parenthesis(sent, ds))

        return predictions

    def init_params(self) -> None:

        if not os.path.exists(Model3.MODEL_PARAMS):
            self.__generate_params()
        else:
            with open(Model3.MODEL_PARAMS, "r") as f:
                self.datasets = set([l.strip() for l in f.readlines()])

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
        ]

        for f in mapfilters:
            ssai_par_datasets = f(ssai_par_datasets)

        mapfilters = [
            MapFilter_PartialMatchDatasets(ssai_par_datasets, Model3.BR_PAT),
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
        train_datasets = [Model3.BR_PAT.sub("", ds).strip() for ds in train_labels_set]
        train_datasets = [
            ds for ds in train_labels_set if sum(ch.islower() for ch in ds) > 0
        ]
        datasets = set(ssai_par_datasets) | set(train_datasets)

        self.datasets = datasets

        with open(Model3.MODEL_PARAMS, "w") as f:
            f.write("\n".join(datasets))

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
                with open(
                    os.path.join(Model3.MODEL_DATA_DIR, "train", (idx + ".json"))
                ) as fp:
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
        for test_file in Path(os.path.join(Model3.MODEL_DATA_DIR, "test")).glob(
            "*.json"
        ):
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
        connection_words = {"of", "the", "with", "for", "in", "to", "on", "and", "up"}

        def map_f(ds):
            toks_spans = list(tokenize_pattern.finditer(ds))
            toks = [t.group() for t in toks_spans]
            start = 0
            if len(toks) > 3:
                if toks[1] == "the":
                    start = toks_spans[2].span()[0]
                elif (
                    toks[0] not in keywords
                    and toks[1] in connection_words
                    and len(toks) > 2
                    and toks[2] in connection_words
                ):
                    start = toks_spans[3].span()[0]
                elif toks[0].endswith("ing") and toks[1] in connection_words:
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
        miss_intro_pat = re.compile("^[A-Z][a-z']+ (?:the|to the) ")
        map_f = lambda ds: miss_intro_pat.sub("", ds)

        super().__init__(map_f)


class MapFilter_BRLessThanTwoWords(MapFilter):
    def __init__(self, br_pat, tokenize_pat):

        filter_f = lambda ds: len(tokenize_pat.findall(br_pat.sub("", ds))) > 2

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

        filter_f = lambda ds: not any(
            (ds in ds_) and (ds != ds_) for ds_ in golden_ds_with_br
        )

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
        (
            tr_counts,
            data_counts,
        ) = MapFilter_TrainCounts.get_train_predictions_counts_data(
            texts,
            MapFilter_TrainCounts.extend_paranthesis(set(datasets)),
            index,
            kw,
            tokenize_pat,
        )
        stats = {}

        for ds, count in Counter(datasets).most_common():
            stats[ds] = [
                count,
                tr_counts[ds],
                tr_counts[re.sub("[\s]?\(.*\)", "", ds)],
                data_counts[ds],
                data_counts[re.sub("[\s]?\(.*\)", "", ds)],
            ]

        def filter_f(ds):
            count, tr_count, tr_count_no_br, dcount, dcount_nobr = stats[ds]
            return (tr_count_no_br > min_train_count) and (
                dcount_nobr / tr_count_no_br > rel_freq_threshold
            )

        super().__init__(filter_f=filter_f)

    @staticmethod
    def extend_paranthesis(datasets):
        # Return each instance of dataset from datasets +
        # the same instance without parenthesis (if there are some)
        pat = re.compile("\(.*\)")
        extended_datasets = []
        for ds in datasets:
            ds_no_parenth = pat.sub("", ds).strip()
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


if __name__ == "__main__":
    input_json_image = sys.argv[1]
    with open(input_json_image, "r") as f:
        text = json.load(f)

    predictions = set(Model3().predict(text))

    print(
        "Model 3 dataset candidates:",
        predictions if len(predictions) else "None found.",
    )
