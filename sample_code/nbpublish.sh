#! /usr/bin/env bash

FILENAME=$1
HTML=$2

if [ "$HTML" = "html" ]
then
    jupyter nbconvert $FILENAME --to html --output-dir="html_from_jupyter"
fi
jupyter nbconvert $FILENAME --to notebook --ClearOutputPreprocessor.enabled=True --output $FILENAME

