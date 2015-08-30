__author__ = 'alexis'

import pandas as pd
import re

train = pd.read_csv('data/train.csv')

def get_title(name):
    s = re.search('([A-Za-z]+)\.', name)

    if s:
        return s.group(1)
    else:
        return ""

def get_family(name):
    s = re.search('([A-Za-z]+)\,', name)

    if s:
        return s.group(1)
    else:
        return ""

titles = train["Name"].apply(get_title)

print set(titles)

family_names = train["Name"].apply(get_family)

print set(family_names)



