#!/bin/bash

echo "arxiv -> Show US Your Data"

python arxiv_helper.py $1

texfiles=($(ls $1/*.tex))

# https://www.tutorialkart.com/bash-shell-scripting/bash-array-length
len=${#texfiles[@]}

if [ $len == 1 ]; then
    fname=${texfiles[0]}
else
    echo "Found $len .tex files:"
    v=1

    for f in ${texfiles[@]}; do
        echo "$v. $f"
        ((v=v+1))
    done

    read -p "Which one should be used:" index
    fname=${texfiles[index-1]}

fi

echo "Converting $fname to XML"

latexml --quiet --dest="$1/$1.xml" $fname

echo "File converted to $1/$1.xml"
echo "Converting to JSON"

python parse_file.py  "$1/$1.xml"

echo "Classifying with Show Us Your Data classifier"

python model_3.py "$1/$1.json"




