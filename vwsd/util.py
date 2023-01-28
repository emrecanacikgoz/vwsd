import os.path as osp
import numpy as np
from random import sample
import sys
import requests
import urllib
import string
from pprint import pprint
import re
from tqdm import tqdm
from time import sleep
import json

current_relations = ['DerivedFrom', 'EtymologicallyDerivedFrom', 'ExternalURL', 'FormOf', 'HasContext', 'IsA', 'RelatedTo', 'Synonym']
blacklist_relations = ['DerivedFrom', 'EtymologicallyDerivedFrom', 'ExternalURL']
whitelist_relations = ['IsA', 'RelatedTo']
printable = set(string.printable)

# regexes
date_re = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}", re.IGNORECASE)
whitespace_re = re.compile(r"[\s\n]+", re.IGNORECASE)
citation_re = re.compile(r'[A-Za-z]+[\s]+[A-Za-z]+[\s]+(\d{4})[\s]+(Cited\sby)?\s+\d+', re.IGNORECASE)
http_re = re.compile(r'http\S+', re.IGNORECASE) 
three_dots_re = re.compile(r'[\S]+\.\.\.', re.IGNORECASE)
wiki_phoneme_re = re.compile(r'\(.*\)', re.IGNORECASE)

def clean_text(info):
    # there are a lot of dates, remove them
    info = date_re.sub("", info) 
    
    # remove punctuation from the data, I think we won't need'em
    info = info.translate(str.maketrans('', '', string.punctuation))

    # remove unprintable characters -> NO PROBLEM, WE TRANSLATED ALL MULTILINGUAL TEXT IN TO ENGLISH
    info = ''.join(filter(lambda x: x in printable, info))
    
    # remove citations
    info = citation_re.sub("", info)

    # replace multiple whitespaces with a single space
    info = whitespace_re.sub(" ", info)

    # remove links
    info = http_re.sub('', info)

    # remove Wikipedia phoneme letters
    info = wiki_phoneme_re.sub('', info)

    # remove three dots at the end of each search result
    info = three_dots_re.sub('', info)
        
    # lower the text
    info = info.lower()
    return info

'''
Convert natural language context to ConceptNet usable context
'''
def prepare_context(context):
    return "_".join(context.split())

'''
Convert ConceptNet information to text
'''
def cnet2str(obj, nl_context):
    cnet_context = "_".join(nl_context.split())

    is_a = ""
    related_to = ""
    is_a_split = ["is a type of", "is a kind of"]

    info = ""
    for edge in obj["edges"]:
        if ("language" in edge["start"] and edge["start"]["language"]) != "en" or ("language" in edge["end"] and edge["end"]["language"] != "en"):
            continue

        # do not use blacklisted edges
        if edge["rel"]["label"] in blacklist_relations:
            continue

        # if the label is "IsA", use it
        if edge["rel"]["label"] == "IsA":
            if "surfaceText" in edge and edge["surfaceText"] is not None:
                # get the context and the its sense, replace with the context in the original surface text
                surf_txt = edge["surfaceText"]
                if "sense_label" in edge["start"]:
                    start_label, start_sense_label = edge["start"]["label"], " ".join(edge["start"]["sense_label"].split()[1:])
                    surf_txt = surf_txt.replace(start_label, start_label + " " + start_sense_label)

                # some text formatting
                surf_txt = " " +  re.sub(r"(\[\[|\]\])", "", surf_txt)
                if is_a:
                    for delimiter in is_a_split:
                        if delimiter in surf_txt:
                            first_half = surf_txt[:surf_txt.index(delimiter)].strip()
                            second_half = surf_txt[surf_txt.index(delimiter)+len(delimiter):].strip()
                            insert = second_half if second_half not in nl_context else first_half
                            is_a += f", {insert}"
                else:
                    is_a += surf_txt

        elif edge["rel"]["label"]  == "RelatedTo":
            start_label, end_label = edge["start"]["label"], edge["end"]["label"]
            if not related_to:
                related_to += f"{nl_context} related to "
            if start_label == cnet_context:
                related_to += end_label + " , "
            else:
                related_to += start_label + " , "
                
    related_to = related_to[:-3] # remove the last comma
                
    info = related_to + " " + is_a
    return info.strip()

def process_path(path):
    return osp.abspath(osp.expanduser(path))


def write_results(results, path):
    path = process_path(path)
    image_files = []
    for batch in results:
        image_files.extend(batch['image_files'])
    with open(path, 'w') as f:
        for this in image_files:
            line = ' '.join(this) + '\n'
            f.write(line)
