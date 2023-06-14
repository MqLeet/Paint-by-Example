#!/bin/bash
main_directory="/data/hongyan/CIHP_PGN/output/dmd_ids/"


for subdir in "$main_directory"/*; do
    if [ -d "${subdir}" ]; then
        DATASET=$(basename "${subdir}")
        python mask_preprocess.py --DATASET "${DATASET}"
    fi
done
