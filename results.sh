#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {25..27}
do
    echo processing day: $i
  
    DAY=$([ $i -lt 10 ] && echo "0$i" || echo "$i")


    gsutil -m cp -R gs://photo-capture-11c8e.appspot.com/subway-subcentro/2023-11-$DAY ./data/pending
    wait


    python3 bytetrack.py -i data/pending/2023-11-$DAY/ -s subway-subcentro -o data/results/sub-centro/
    wait

    rm -r data/pending/2023-11-$DAY

    trash-empty
done
