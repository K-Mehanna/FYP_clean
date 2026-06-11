import pandas as pd
import numpy as np
import math
import sys
from pathlib import Path

def load_eortc(BASE_PATH: Path, BRAINWEAR_FILE: str):
    filepath = BASE_PATH/BRAINWEAR_FILE
    eortc = load_pro_data(filepath, "EORTC QLQ")
    eortc_scores = calculate_eortc_scores(eortc)
    return eortc, eortc_scores


def load_pro_data(filepath: Path, sheet_name: str):
    df = pd.read_excel(filepath, sheet_name)
    df = df.rename(columns=lambda x: x.strip())
    return df


eortc_scoring_table = {
    # qlq_c30 scoring table
    "QL2": {'name': 'Global health status',   'items': 2, 'range': 6, 'questions': [29, 30],         'function_scale': False},
    "PF2": {'name': 'Physical functioning',   'items': 5, 'range': 3, 'questions': [1, 2, 3, 4, 5],  'function_scale': True},
    "RF2": {'name': 'Role functioning',       'items': 2, 'range': 3, 'questions': [6, 7],           'function_scale': True},
    "EF": {'name': 'Emotional functioning',  'items': 4, 'range': 3, 'questions': [21, 22, 23, 24], 'function_scale': True},
    "CF": {'name': 'Cognitive functioning',  'items': 2, 'range': 3, 'questions': [20, 25],         'function_scale': True},
    "SF": {'name': 'Social functioning',     'items': 2, 'range': 3, 'questions': [26, 27],         'function_scale': True},
    "FA": {'name': 'Fatigue',                'items': 3, 'range': 3, 'questions': [10, 12, 18],     'function_scale': False},
    "NV": {'name': 'Nausea and Vomiting',    'items': 2, 'range': 3, 'questions': [14, 15],         'function_scale': False},
    "PA": {'name': 'Pain',                   'items': 2, 'range': 3, 'questions': [9, 19],          'function_scale': False},
    "DY": {'name': 'Dyspnoea',               'items': 1, 'range': 3, 'questions': [8],              'function_scale': False},
    "SL": {'name': 'Insomnia',               'items': 1, 'range': 3, 'questions': [11],             'function_scale': False},
    "AP": {'name': 'Appetite loss',          'items': 1, 'range': 3, 'questions': [13],             'function_scale': False},
    "CO": {'name': 'Constipation',           'items': 1, 'range': 3, 'questions': [16],             'function_scale': False},
    "DI": {'name': 'Diarrhoea',              'items': 1, 'range': 3, 'questions': [17],             'function_scale': False},
    "FI": {'name': 'Financial Difficulties', 'items': 1, 'range': 3, 'questions': [28],             'function_scale': False},
    # bn20 scoring table - can find scoring from https://github.com/cran/QoLR/blob/master/R/scoring.QLQBN20.R
    "BNFU": {'name': 'Future Uncertainty',    'items': 4, 'range': 3, 'questions': [31, 32, 33, 35], 'function_scale': False},
    "BNVD": {'name': 'Visual Disorder',       'items': 3, 'range': 3, 'questions': [36, 37, 38],     'function_scale': False},
    "BNMD": {'name': 'Motor Dysfunction',     'items': 3, 'range': 3, 'questions': [40, 45, 49],     'function_scale': False},
    "BNCD": {'name': 'Communication Deficit', 'items': 3, 'range': 3, 'questions': [41, 42, 43],     'function_scale': False},
    "BNHA": {'name': 'Headaches',             'items': 1, 'range': 3, 'questions': [34],             'function_scale': False},
    "BNSE": {'name': 'Seizures',              'items': 1, 'range': 3, 'questions': [39],             'function_scale': False},
    "BNDR": {'name': 'Drowsiness',            'items': 1, 'range': 3, 'questions': [44],             'function_scale': False},
    "BNIS": {'name': 'Itching Skin',          'items': 1, 'range': 3, 'questions': [47],             'function_scale': False},
    "BNHL": {'name': 'Hair Loss',             'items': 1, 'range': 3, 'questions': [46],             'function_scale': False},
    "BNWL": {'name': 'Weakness of Legs',      'items': 1, 'range': 3, 'questions': [48],             'function_scale': False},
    "BNBC": {'name': 'Bladder Control',       'items': 1, 'range': 3, 'questions': [50],             'function_scale': False},
}


def calculate_eortc_scores(eortc: pd.DataFrame) -> pd.DataFrame:

    def functional_linear_transform(raw_score, score_range):
        return (1-(raw_score-1)/score_range) * 100

    def other_linear_transform(raw_score, score_range):
        return ((raw_score-1)/score_range) * 100

    # Calculate scores
    eortc_scores = pd.DataFrame(columns=eortc.columns)

    for key, value in eortc_scoring_table.items():
        values = eortc[eortc['Question'].isin(value['questions'])].copy()
        for column in values.columns:
            if values[column].count() < (len(values) / 2):
                values[column] = np.NaN
        raw_scores = values.mean()
        transform = functional_linear_transform if value['function_scale'] else other_linear_transform
        scores = raw_scores.apply(transform, args=(value['range'],))
        scores = scores.astype(object) # Cast to object to allow strings - removes warning
        scores.iloc[0] = key
        scores = pd.DataFrame(scores).T
        eortc_scores = pd.concat([eortc_scores, scores], axis=0)

    eortc_scores = pd.concat(
        [pd.DataFrame(eortc.iloc[0]).T, eortc_scores], axis=0)

    eortc_scores = eortc_scores.reset_index(drop=True)
    return eortc_scores

 