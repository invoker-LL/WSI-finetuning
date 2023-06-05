CUDA_VISIBLE_DEVICES=4 python train_end2end.py --exp_code 'FT_Camelyon16_res50_fc_freeze_layer0_decay1e-3_ema997' --k_start 0 --k_end 5 --bag_size 512 \
--data_root_dir './data_feat/Camel16_ostu_top512_vib' --model_size 'small' --ema_decay 0.997
