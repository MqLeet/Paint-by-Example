export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="/data/hongyan/Paint-by-Example/ldm:$PYTHONPATH"
python ./scripts/inference_folder.py \
--plms --outdir ./results_hy_test/outpainted_images_0.7/yawning_and_sleepy/ \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--seed 325 \
--scale 5 \
--image_dir /data/hongyan/CIHP_PGN/datasets/outpainted_images_0.7/images/ \
--mask_dir /data/hongyan/CIHP_PGN/output/outpainted_images_0.7/edit_mask/ \
--reference_dir /home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/mapper/yawning_and_sleepy/inference_results/ 
# /home/duanyuxuan/dms_gen/CIHP_PGN/datasets/dmd_extract1k_crop768-1024/images/     \
# /home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/edit_mask/       \


#/home/duanyuxuan/dms_gen/CIHP_PGN/output/dmd_extract1k_crop768-1024/cihp_parsing_maps/ 
#/home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/mapper/yawning-sleepy/inference_results/ 

# path：/home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/mapper/yawn/inference_results_0.75
# /home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/mapper/yawning_and_sleepy/inference_results
# /home/duanyuxuan/dms_gen/StyleCLIP/mapper/results/mapper/yawning-sleepy