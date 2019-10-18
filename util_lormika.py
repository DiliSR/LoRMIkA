#!/usr/bin/env python3
import numpy as np
import pandas as pd
import _pickle as cPickle
import random

from sklearn.preprocessing import LabelEncoder


def label_encode(df, columns, label_encoder=None):
    df_le = df.copy(deep=True)
    new_le = label_encoder is None
    label_encoder = dict() if new_le else label_encoder
    for col in columns:
        if new_le:
            le = LabelEncoder()
            df_le[col] = le.fit_transform(df_le[col])
            label_encoder[col] = le
        else:
            le = label_encoder[col]
            df_le[col] = le.transform(df_le[col])
    return df_le, label_encoder


def label_decode(df, columns, label_encoder):
    df_de = df.copy(deep=True)
    for col in columns:
        le = label_encoder[col]
        df_de[col] = le.inverse_transform(df_de[col])
    return df_de


def remove_variables(df, vars_to_remove):
    df.drop(vars_to_remove, axis=1, inplace=True, errors='ignore')
