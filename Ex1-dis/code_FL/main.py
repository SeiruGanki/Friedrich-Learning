from train_phase1 import train1
from train_phase2 import train2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pretrain_num = 1000

lr_u = 3e-4
lr_v = 3e-3
exp_bench_u = 9000
exp_bench_v = 9000

train1(lr_u,lr_v,exp_bench_u,exp_bench_v,pretrain_num)
train2(lr_u,lr_v,exp_bench_u,exp_bench_v,pretrain_num)