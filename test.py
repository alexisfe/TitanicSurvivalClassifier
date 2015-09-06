__author__ = 'alexis'

import pandas as pd
import re
import numpy as np

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

# titles = train["Name"].apply(get_title)
#
# print set(titles)
#
# family_names = train["Name"].apply(get_family)
#
# print set(family_names)

#t = train["Ticket"].value_counts()
# #print t
# ticket = '1601'
#
# print train.loc[train["Ticket"] == ticket, "Fare"]
#
# t = train.loc[train["Ticket"] == ticket, "Fare"]/train.loc[train["Ticket"] == ticket, "Ticket"].count()
#
# train.loc[train["Ticket"] == ticket, "Fare"] = t.values
#
# print train.loc[train["Ticket"] == ticket, "Fare"]

t = train["Ticket"].value_counts()
t = t[t.values > 1]



