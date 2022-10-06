# file https://github.com/Coleridge-Initiative/rc-kaggle-models/tree/original_submissions/4th%20OsciiArt%20resistance0108%20Naoto_Usuyama/Kaggle_Coleridge_4th_Solution-main
# notebooks:
# submission 1: https://www.kaggle.com/code/osciiart/210622-det1-neru-train-govt/notebook?scriptVersionId=66367000 Public: 0.614, Private: 0.513
# submission 2: https://www.kaggle.com/osciiart/210621-train-acronym-ver1?scriptVersionId=66241779 (broken link) Public: 0.622, Private: 0.493

import re
from typing import Dict, List

import spacy


from model import Model



class Model4(Model):
    MODEL_PATH = "model4/spacynerlexeme0620/model-best"
    KEYWORDS = [
        'study',
        'studies',
        'data',
        'survey',
        'panel',
        'census',
        'cohort',
        'longitudinal',
        'registry',
    ]

    KEYWORDS2 = [
        'study',
        'studies',
        'data',
        'survey',
        'panel',
        'census',
        'cohort',
        'longitudinal',
        'registry',
        'the',
    ]

    KEYWORDS3 = [
        'study',
        'studies',
        'dataset',
        'database',
        'survey',
        'panel',
        'census',
        'cohort',
        'longitudinal',
        'registry',
    ]

    KEYWORDS4 = [
        'system',
        'center',
        'centre',
        'committee',
        'documentation',
        'entry',
        'assimilation',
        'explorer',
        'regulation',
        'portal',
        'format',
        'data science',
        'analysis',
        'management',
        'agreement',
        'branch',
        'acquisition',
        'request',
        'task force',
        'program',
        'operator',
        'office',
        'data view',
        'data language',
        'mission',
        'alliance',
        'data model',
        'data structure',
        'corporation',
    ]

    KEYWORDS5 = KEYWORDS + ['of', 'the', 'national', 'education']

    WHITE_LIST = [
        'ipeds',
    ]

    NG_LIST = [
        'national longitudinal survey',
        'education longitudinal survey',
        'census bureau',
        'data appendix',
        'data file user',
        'supplementary data',
        'data supplement',
        'major field of study'
    ]

    BLACK_LIST = [
        'USGS',
        'GWAS',
        'ECLS',
        'DAS',
        'NCDC',
        'NDBC',
        'UDS',
        'GTD',
        'ISC',
        'DGP',
        'EDC',
        'FDA',
        'TSE',
        'DEA',
        'CDA',
        'IDB',
        'NGDC',
        'JODC',
        'EDM',
        'FADN',
        'LRD',
        'DBDM',
        'DMC',
        'WSC',
    ]

    def __init__(self) -> None:

        self.ner_model = spacey.load(Model4.MODEL_PATH)


        super().__init__()

    def preprocess(self, json_text: List[Dict[str, str]]) -> str:

        text = " ".join(list(map(lambda x: x["text"], json_text)))
        cleaned_text = clean_text(text)
        # dup_id are training documents that match
        # dup is a bool that indicates the test text is found in the training text
        #    the first  document is manually set to false -- why?
        # df_label_train == df_label2
        # det_train


        # df_test_reduct
        # TODO: continue here: https://www.kaggle.com/code/osciiart/210622-det1-neru-train-govt/notebook?scriptVersionId=66367000#Test-prediction

        return super().preprocess(json_text)


    def predict(self, json_text: str) -> List[str]:

        processed = self.preprocess(json_text)

        datasets = [""]

        return datasets


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


