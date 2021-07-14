#!/bin/bash
reset;
source ~/anaconda/etc/profile.d/conda.sh;
conda activate SegNetsTF2;
python main.py --phase test --model_dir ./checkpoint/segSimonRock_BIN_180_L1-10.0_sr-False_c1-False_ac-True_c2-False


