__author__ = 'alexis'

import re
import math

from sklearn.preprocessing import LabelEncoder

default_titles = ["Mr", "Mrs", "Ms", "Miss", "Mx", "Master", "Maid", "Madam", "Madame", "Mlle", "Mme"]
academic_titles = ["Dr"]
aristocratic_titles = ["Hon", "Don", "Lady", "Countess", "Jonkheer", "Sir"]
military_titles = ["Major", "Col", "Capt"]
religious_titles = ["Rev"]

def get_ticket_class(ticket):
    ticket_search = re.search('[Aa](\.)*', ticket)

def get_family_name(name):
    family_search = re.search('([A-Za-z]+)\,', name)
    if family_search:
        return family_search.group(1)
    return ""

def get_cabin_type(cabin):
    if type(cabin) is float:
        if math.isnan(cabin):
            return "Z"

    return str(cabin)[0]

def get_num_cabins(cabin):
    if type(cabin) is float:
        if math.isnan(cabin):
            return 1

    return len(re.findall('[A-Z]', str(cabin)))

def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def get_num_tickets(ds):
    return
    #return ds.loc[ds["Ticket"] == ticket, "Ticket"].count()

def get_right_fare(ds):
    #Tickets frequency table
    ts = ds["Ticket"].value_counts()
    #Extract duplicated tickets
    ts = ts[ts.values > 1]

    for ticket in ts.index:
        #Calculate right fare
        rf = ds.loc[ds["Ticket"] == ticket, "Fare"]/ds.loc[ds["Ticket"] == ticket, "Ticket"].count()
        #Replace with right fare
        ds.loc[ds["Ticket"] == ticket, "Fare"] = rf.values

    return ds["Fare"]

def encode(ds):
    le = LabelEncoder()
    le.fit(ds)
    return le.transform(ds)

def transform(ds, age_median):
    ds['Age'] = ds['Age'].fillna(age_median)

    ds["Sex"] = encode(ds["Sex"])

    ds['Fare'] = ds['Fare'].fillna(ds['Fare'].median())

    #Transform fare for people with the same ticket number
    ds['Fare'] = get_right_fare(ds[['Ticket', 'Fare']])

    ds['Embarked'] = ds['Embarked'].fillna('S')
    ds["Embarked"] = encode(ds["Embarked"])

    ds["FamilySize"] = ds["SibSp"] + ds["Parch"]

    #Calculate name features
    ds["NameLength"] = ds["Name"].apply(lambda x: len(x))

    titles = ds["Name"].apply(get_title)
    ds["Title"] = encode(titles)

    family_names = ds["Name"].apply(get_family_name)
    ds["FamilyName"] = encode(family_names)

    #Calculate cabin features
    cabin_types = ds["Cabin"].apply(get_cabin_type)
    ds["CabinType"] = encode(cabin_types)

    ds["NumCabins"] = ds["Cabin"].apply(get_num_cabins)

    #Calculate ticket features
    #ds["NumTickets"] = ds.apply(get_num_tickets, axis=1)

    return ds
