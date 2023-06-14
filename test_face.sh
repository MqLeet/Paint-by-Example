python scripts/inference.py \
--plms --outdir results_face \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--seed 325 \
--scale 5 \
--image_path /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/104_2085.jpg \
--mask_path /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/104_2085_vis.png \
--reference_path /home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/sleepy/00019_sleepy.jpg


python scripts/inference.py \
--plms --outdir results_face \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--seed 326 \
--scale 5 \
--image_path /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/104_2085.jpg \
--mask_path /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/104_2085_vis.png \
--reference_path /home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/sleepy/00041_sleepy.jpg \



# python scripts/inference.py \
# --plms --outdir results_face \
# --config configs/v1.yaml \
# --ckpt checkpoints/model.ckpt \
# --seed 324 \
# --scale 5 \
# --image_path /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/104_2085.jpg \
# --mask_path /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/104_2085_vis.png \
# --reference_path /home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/yawn/inference_results_0.5/00003.jpg


# python scripts/inference.py \
# --plms --outdir results_face \
# --config configs/v1.yaml \
# --ckpt checkpoints/model.ckpt \
# --seed 321 \
# --scale 5 \
# --image_path /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/104_2085.jpg \
# --mask_path /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/104_2085_vis.png \
# --reference_path /home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/drunk/inference_results/00005.jpg \


# python scripts/inference.py \
# --plms --outdir results_face \
# --config configs/v1.yaml \
# --ckpt checkpoints/model.ckpt \
# --seed 322 \
# --scale 5 \
# --image_path /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/104_2085.jpg \
# --mask_path /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/104_2085_vis.png \
# --reference_path /home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/drunk/inference_results/00009.jpg
