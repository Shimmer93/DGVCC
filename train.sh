# python main.py --train --config configs/sta_reg.yml --device cuda:1 --only_reg
# python main.py --train --config configs/sta_dg.yml --device cuda:3 --ckpt logs/sta_reg4/best.pth
# python main_dg.py --task vis --config configs/sta_reg.yml
# python main_adv.py --task train --config configs/sta_joint.yml
python main_adv.py --task vis --config configs/sta_final.yml