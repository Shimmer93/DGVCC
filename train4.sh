# python main.py --train --config configs/sta_reg.yml --device cuda:1 --only_reg
# python main.py --train --config configs/sta_dg.yml --device cuda:3 --ckpt logs/sta_reg4/best.pth
# python main_dg.py --task vis --config configs/sta_reg.yml
# python main_adv.py --task vis --config configs/sta_final.yml
# python main_cls.py --task vis --config configs/sta_reg3.yml
# python main_cls.py --task train --config configs/sta_gen2.yml
# python main_cls.py --task vis --config configs/sta_final2.yml
CUDA_VISIBLE_DEVICES=3 python main.py --task train_test --config configs/stb_final10.yml
CUDA_VISIBLE_DEVICES=3 python main.py --task train_test --config configs/stb_final11.yml