#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse, configparser, datetime, email, email.parser, glob, os, pathlib, pickle, re, sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup, UnicodeDammit

import common
from common import XDG_CONFIG_HOME, XDG_DATA_HOME


class NormPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, common.norm_path(values))


def parse_args():
    parser = argparse.ArgumentParser(description='Reads an email from stdin, classifies it and outputs to stdout.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", help="Action to take. /classify/ mail delivered on stdout. /rebuild/ training data.",
                        choices = ["classify", "rebuild"], nargs = "?", default = "classify")
    parser.add_argument("--config", help="Path to config file.",
                        default = XDG_CONFIG_HOME("flofify", "config"), action=NormPath)
    parser.add_argument("--model", help="Model file to use.",
                        default = XDG_DATA_HOME("flofify", "model"), action=NormPath)
    parser.add_argument("--vocabulary", help="Vocabulary file to use.",
                        default = XDG_DATA_HOME("flofify", "vocabulary"), action=NormPath)

    return parser.parse_args()

    
class Bucket():
    def __init__(self, name, **args):
        self.name = name
        self.patterns = args["patterns"]
        self.min_probability = float(args["min_probability"])
        self.max_age = args["max_age"] # Maximum age of messages that are learned form. datetime.timedelta object
        
    def __repr__(self):
        return self.name

    def __str__(self):
        return "Bucket: %s, Size: %s elements" % (self.name, len(self.files()))    

    def files(self):
        """ Returns a list of files that matches the pattern set in constructor. """
        fs = []
        for pattern in self.patterns.split(":"):
            fs += glob.glob(pattern)
        print("Mails before filtering in bucket:", self.name, "Count:", len(fs))
        filtered = [f for f in fs if self.check_age(f)]
        print("Mails after filtering in bucket:", self.name, "Count:", len(filtered))
        return filtered

    def check_age(self, file):
        """ Checks if a message is older than self.max_age. Returns False if so. """
        fp = open(file, "r", errors = "ignore")
        try:
            msg = email.parser.Parser().parse(fp, headersonly=True)
        except UnicodeDecodeError as e: 
            print("Error reading file", file, "Ignoring file. Error is:")
            print(e)
            return False
        # datetime.strptime("Sat, 16 Aug 2014 16:26:14 +0400", "%a, %d %b %Y %H:%M:%S %z")
        datestr = msg.get_all("Received")[0].split(";")[-1].lstrip()
        datestr = datestr[:31].strip() # Only pick string up to +0400, ignore a (CEST) if existing
        try:
            dtime = datetime.datetime.strptime(datestr, "%a, %d %b %Y %H:%M:%S %z")
        except TypeError:
            print("Error converting date in", file)
            return False

        if datetime.datetime.now(datetime.timezone.utc) - dtime > self.max_age:
            return False
        else:
            return True
        

    def train_data(self):
        """ Returns a numpy array of shape (n, 2). ID (name of bucket) in first row, filenames in second row."""
        files = self.files()
        print("Got", len(files), "files for learning.")
        data = np.array( [[self.name]*len(files), files] )
        return data.transpose()


class Model:
    PICKLE_PROTOCOL = 2

    def __init__(self, buckets, fields):
        self.buckets = buckets
        self.fields = fields

    
    def train(self):
        vectorizer = CountVectorizer(input='filename', decode_error='replace', strip_accents='unicode',
                                     preprocessor=self.mail_preprocessor, stop_words='english', max_df = 0.8)
        transformer = TfidfTransformer()
        self.classifier = MultinomialNB()

        data = np.vstack( [i.train_data() for i in self.buckets] )
        vectors = vectorizer.fit_transform(data[:,1])
        X = transformer.fit_transform(vectors)
        y = data[:,0]
        
        self.classifier.fit(X, y)
        self.vocabulary = vectorizer.vocabulary_
        print("Learned from %s mails." % data.shape[0])
        for b in self.buckets:
            print(b)
        print("Fields:  %s" % self.fields)
        
        
    def classify(self, text):
        """ Classsifies text, returns tuple (final class, probability, class). if probability is larger than min_probability then final class == class"""
        vectorizer = CountVectorizer(input="content", vocabulary=self.vocabulary, decode_error='replace', strip_accents='unicode',
                                     preprocessor=self.mail_preprocessor, stop_words='english')
        transformer = TfidfTransformer()
        vectors = vectorizer.transform( [text] )
        X = transformer.fit_transform(vectors)
        proba = np.max(self.classifier.predict_proba(X))
        c = self.classifier.predict(X)[0]
        bucket = next( (b for b in self.buckets if b.name == c) )
        if proba >= bucket.min_probability:
            return (c, proba, c)
        else:
            return (None, proba, c)        

        
    def save(self, model_path, vocabulary_path):
        model = open(model_path, "wb")
        pickle.dump(self.classifier, model, self.PICKLE_PROTOCOL)

        voc = open(vocabulary_path, "wb")
        pickle.dump(self.vocabulary, voc, self.PICKLE_PROTOCOL)
        

    def load(self, model_path, vocabulary_path):
        model = open(model_path, "rb")
        self.classifier = pickle.load(model)

        voc = open(vocabulary_path, "rb")
        self.vocabulary = pickle.load(voc)
        
    def mail_preprocessor(self, message):
        """ Extracts the text. Combines body, From and Subject headers."""
        # Filter POPFile cruft by matching date string at the beginning.
        pop_reg = re.compile(r"^[0-9]{4}/[0-1][1-9]/[0-3]?[0-9]")
        message = [line for line in message.splitlines(True) if not pop_reg.match(line)]
        msg = email.message_from_string("".join(message))
        
        msg_body = ""

        if "body" in self.fields:
            for part in msg.walk():
                if part.get_content_type() in ["text/plain", "text/html"]:
                    body = part.get_payload(decode=True)
                    soup = BeautifulSoup(body)
                    msg_body += soup.get_text(" ", strip=True)

            """ Ignore encrypted messages. """
            if "-----BEGIN PGP MESSAGE-----" in msg_body:
                msg_body = ""
                
        if "from" in self.fields:
            msg_body += " ".join(email.utils.parseaddr(msg["From"]))

        if "subject" in self.fields:
            try:
                msg_body += " " + msg["Subject"]
            except TypeError: # Can't convert 'NoneType' object to str implicitly
                pass
                
        msg_body = msg_body.lower()
        return msg_body

        
    
    
class Configuration(configparser.ConfigParser):
    def buckets(self):
        """ Returns a list of buckets created from every config section that starts with "Bucket:", with the leading "Bucket:" cut out from the buckets name."""

        bs = []
        for s in self.sections():
            if s.startswith("Bucket:"):
                name = s[7:]
                min_prob = self.min_probability(s)
                max_age = self.max_age(s)
                bs.append(Bucket(name, patterns = self[s]["patterns"], min_probability = min_prob, max_age = max_age))
            
        return bs

    def min_probability(self, section):
        return float(self[section]["min_probability"])
                      
    def max_age(self, section):
        # datetime.strptime("Sat, 16 Aug 2014 16:26:14 +0400", "%a, %d %b %Y %H:%M:%S %z")
        max_age = self[section].get("max_age", -1)
        if max_age == -1:
            return datetime.timedelta.max
        else:
            return datetime.timedelta(days = int(max_age))
                    
    def default_bucket(self):
        return self["Global"].get("default_bucket", "None")

    def fields(self):
        """ Get the fields that are to be taken into account. """    
        default_fields = "From Subject Body"    
        f = self["Global"].get("fields", default_fields)
        f = f.lower()
        return [ i.strip() for i in f.split(" ") ]

        

def main():
    args = parse_args()

    pathlib.Path(args.config).parent.mkdir(parents = True, exist_ok = True)
    pathlib.Path(args.model).parent.mkdir(parents = True, exist_ok = True)
    pathlib.Path(args.vocabulary).parent.mkdir(parents = True, exist_ok = True)
    
    config = Configuration()
    config.read(args.config)

    model = Model(config.buckets(), config.fields())
    if args.action == "rebuild":
        model.train()
        model.save(args.model, args.vocabulary)
    elif args.action == "classify":
        mail = UnicodeDammit(sys.stdin.detach().read())
        if mail:
            model.load(args.model, args.vocabulary)
            classification = model.classify(mail.unicode_markup)
            msg = email.message_from_string(mail.unicode_markup)
            if classification[0] == None:
                msg["X-Flofify-Class"] = config.default_bucket()
            else:
                msg["X-Flofify-Class"] = str(classification[0])
            
            msg["X-Flofify-Probability"] = str(round(classification[1], 4)) + ", " + str(classification[2])
            sys.stdout.write(msg.as_string())
        else:
            sys.stdout.write(mail.unicode_markup)




if __name__ == "__main__":
    main()
