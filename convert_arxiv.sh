#!/bin/bash

echo "arxiv -> Show US Your Data Tool"

python arxiv_helper.py $1

texfiles=($(ls $1/*.tex))

# https://www.tutorialkart.com/bash-shell-scripting/bash-array-length/#:~:text=To%20get%20length%20of%20an%20array%20in%20Bash%2C,returns%20the%20number%20of%20elements%20in%20the%20array.
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




