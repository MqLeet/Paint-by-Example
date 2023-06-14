#!/bin/bash
export PYTHONPATH="/data/hongyan/Paint-by-Example/ldm:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1,2
input_folder="/data/hongyan/CIHP_PGN/output/dmd_ids/"
imagedir_base="/data/hongyan/CIHP_PGN/datasets/dmd_ids/"
reference_dir="/home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/mapper/yawning_and_sleepy/inference_results/"
outdir_base="./results_hy_test/dmd_ids/yawning_and_sleepy/"

for d in "${input_folder}"/*/; do
    subfolder_name=$(basename "${d}")
    outdir="${outdir_base}${subfolder_name}/"
    image_dir="${imagedir_base}${subfolder_name}/images/"
    mask_dir="${d}/edit_mask/"
    if [ ! -d "${outdir}" ]; then
        python ./scripts/inference_folder.py \
            --plms --outdir "${outdir}" \
            --config configs/v1.yaml \
            --ckpt checkpoints/model.ckpt \
            --seed 325 \
            --scale 5 \
            --image_dir "${image_dir}" \
            --mask_dir "${mask_dir}" \
            --reference_dir "${reference_dir}"
    else
        echo "Skipping ${outdir} as it already exists"
    fi
done
