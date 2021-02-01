#!/bin/bash

# files for io_tutorial.ipynb
wget 'https://docs.google.com/uc?export=download&id=1B-xX3F65JcWzAg0v7f1aVwnawPIfb5_o' -O sample_code/data/small4DSTEMscan_10x10.dm3
wget 'https://docs.google.com/uc?export=download&id=12Q3T57x9N2vkyY0llqBLKn_0JPurQM6Y' -O sample_code/data/small4DSTEMscan_10x10.h5

# files for classification_twinBoundary.ipynb
# long winded but allows for downloading "large" (>100mb) from Gdrive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sUrPEgM1wWyTh-LJ30lGUhcXklHj6ajC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sUrPEgM1wWyTh-LJ30lGUhcXklHj6ajC" -O sample_code/data/twinBoundary_ShitengZhao20190115MEA.h5 && rm -rf /tmp/cookies.txt
