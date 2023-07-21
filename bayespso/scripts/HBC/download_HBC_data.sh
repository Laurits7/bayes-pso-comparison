#!/bin/bash

URL=http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz
FILENAME=atlas-higgs-challenge-2014-v2.csv.gz

curl "$URL" "$FILENAME"
gunzip "$FILENAME"
