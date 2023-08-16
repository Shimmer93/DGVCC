# python main.py --train --config configs/sta_reg.yml --device cuda:1 --only_reg
# python main.py --train --config configs/sta_dg.yml --device cuda:3 --ckpt logs/sta_reg4/best.pth
# python main_adv.py --task vis --config configs/sta_joint.yml
# python main_adv.py --task test --config configs/sta_afterjoint.yml
CUDA_VISIBLE_DEVICES=1 python main.py --task train_test --config configs/stb_final4.yml
CUDA_VISIBLE_DEVICES=1 python main.py --task train_test --config configs/stb_final5.yml
CUDA_VISIBLE_DEVICES=1 python main.py --task train_test --config configs/stb_final6.yml