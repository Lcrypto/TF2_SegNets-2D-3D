#!/bin/bash
reset;
source ~/anaconda/etc/profile.d/conda.sh;
conda activate SegNetsTF2;
python main.py --phase train --fine_size 180;

# --fine_size  X
# size X not larger than smalles of downsampled 4 times input images

#--load_size
# recomend 4 times larger than fine_size  X


# if use grayscale segmentation map use one channel --output_nc 1
# but you can try to train color (rgb) map segmentation as well --output_nc 3
