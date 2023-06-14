

python scripts/inference.py \
--plms --outdir results_dmd \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--seed 321 \
--scale 5 \
--image_path /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/25_2085.jpg \
--mask_path /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/25_2085_vis.png \
--reference_path /data/hongyan/posetransfer/Pose-Transfer/imgs/women1.jpg \


python scripts/inference.py \
--plms --outdir results_dmd \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--seed 322 \
--scale 5 \
--image_path /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/25_2085.jpg \
--mask_path /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/25_2085_vis.png \
--reference_path /data/hongyan/posetransfer/ADGAN/ADGAN/deepfashion/test_high/fashionMENJackets_Vestsid0000065304_7additional.jpg

python scripts/inference.py \
--plms --outdir results_dmd \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--seed 325 \
--scale 5 \
--image_path /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/25_2085.jpg \
--mask_path /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/25_2085_vis.png \
--reference_path /data/hongyan/posetransfer/ADGAN/ADGAN/deepfashion/test_high/fashionMENJackets_Vestsid0000432802_1front.jpg

python scripts/inference.py \
--plms --outdir results_dmd \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--seed 324 \
--scale 5 \
--image_path /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/25_2085.jpg \
--mask_path /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/25_2085_vis.png \
--reference_path /data/hongyan/posetransfer/ADGAN/ADGAN/deepfashion/test_high/fashionMENTees_Tanksid0000012201_2side.jpg
