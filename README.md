# TensorFlow 2.5 SegNets- 2D/3D Deep Neural Network Using Stack of Super Resolution Deep Neural Network and GANs: cycleGAN, SRGAN, ESRGAN  

Ready to use rewritten project from original rep https://github.com/yingDaWang-UNSW/SegNets-3D for doing end-to-end Train and Inference 2D/3D images Super resolution with Segmentation using SRCNN Resnet and SRGANs: ESRGAN, SRGAN, WDSRGAN. Ported to use TensorFlow- 2.5 and tf-slim. For detail read Wang Ying Da PH.D THESIS Machine Learning Methods and Computationally Efficient Techniques in Digital Rock Analysis, 2020 and all related articles. Please put a star to https://github.com/yingDaWang-UNSW to support his work.



Because Dr Ying Da Wang not include train dataset i  add images for train from 'CamVid data set' Brostow, Gabriel J., Julien Fauqueur, and Roberto Cipolla. "Semantic Object Classes in Video: A High-Definition Ground Truth Database." Pattern Recognition Letters. Vol. 30, Issue 2, 2009, pp 88-97.




Preparation of Env for SRDNN Train.


Let consider Mac OS X, another platforms require similar dependency.

Root folder of project './SegNets-3D/' 

1. run sh script  installSRMac.sh

installSRMac.sh 

```
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-MacOSX-x86_64.sh;
bash Anaconda3-2021.05-MacOSX-x86_64.sh -b;

~/anaconda/bin/conda init bash;
source ~/.bash_profile; # just in case;

conda create --name SegNetsTF2;

source ~/anaconda/etc/profile.d/conda.sh;
conda activate SegNetsTF2;
conda info --envs;


conda install tensorflow==2.5.0; 
conda install tensorflow-gpu==2.5.0; 
conda install pillow==8.2.0;
pip install tensorlayer==2.2.23;
pip install tf-slim==1.1.0;
```


2. Convert your train input downsampled 4 times images, segmentation color or grayscale map (with 4 times larger resolution to numpy (or pickle format).
If you don't know how use script runSRpconversion.sh from https://github.com/Lcrypto/EDSRGAN-3D project. if you use 'png' conversion not required.


Place downsampled 4 times images to folder  '/datasets/segSimonRock_BIN/trainA'


Place segmentation color or grayscale map images '/datasets/segSimonRock_BIN/trainB'



3. choice correct input size parameter according smalles size of downsampled image --fine_size 180.  
 ```
 --load_size 4*X  recomend to chice 4 times larger than fine_size  X
```
if use grayscale segmentation map use one channel --output_nc 1,  but you can try to train color (rgb) map segmentation as well --output_nc 3



4. Choice train parameters and architecture by changing defaul  values  at main.py or use arguments
 
 python main.py --phase train --fine_size 180
 
 
```

 training data arguments
parser.add_argument('--gpuIDs', dest='gpuIDs', type=str, default='0', help='IDs for the GPUs. Empty for CPU. Nospaces')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='segSimonRock_BIN', help='path of the dataset')
parser.add_argument('--load2ram', dest='load2ram', type=bool, default=False, help='load dataset into ram')
parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=25, help='# of epoch to decay lr')
parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=2, help='# images used to train per epoch')
parser.add_argument('--iterNum', dest='iterNum', type=str2int, default=50, help='# iterations per epoch')
parser.add_argument('--load_size', dest='load_size', type=int, default=720,
                    help='scale images to this size')  # only active if SC1GAN, this is turned off for C2GAN and ACGAN
parser.add_argument('--fine_size', dest='fine_size', type=str2int, default=180, help='then crop to this size')
parser.add_argument('--nDims', dest='nDims', type=str2int, default=2, help='2D or 3D inputs and outputs')
parser.add_argument('--ngf', dest='ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=8, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3,
                    help='# of image channels for A')  # 1 for 3D, 3 for 2D colour
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3,
                    help='# of image channels for B')  # 1 for seg, 3 for SR etc etc

parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=10, help='save a model every save_freq epochs')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50,
                    help='print the validation images every X epochs')
parser.add_argument('--continue_train', dest='continue_train', type=str2bool, default=False,
                    help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--save_dir', dest='save_dir', default=None,
                    help='models are saved here, if none, will generate based on some params')
parser.add_argument('--model_dir', dest='model_dir', default=None, help='models are loaded from here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='samples are saved here')

parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=False,
                    help='generation network using residual block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=str2bool, default=False,
                    help='gan loss defined in lsgan')  # patchGAN plays poorly with scgan
# symcycleganFlags
parser.add_argument('--c1ganFlag', dest='c1ganFlag', type=bool, default=False,
                    help='flag for training a symmetric type cyclegan network')
# SRGAN arguments
parser.add_argument('--srganFlag', dest='srganFlag', type=bool, default=False,
                    help='flag for training a feed forward network')
# asymCGAN arguments
parser.add_argument('--ACGANFlag', dest='ACGANFlag', type=bool, default=True,
                    help='flag for training the Asymetric cyclegan network')
# C2GAN arguments
parser.add_argument('--c2ganFlag', dest='c2ganFlag', type=bool, default=False,
                    help='flag for training the c2gan network')
# asymmetric models
# parser.add_argument('--acType', dest='acType', type=str, default='semSeg', help='which model is asymetric, semSeg, or superRes')
parser.add_argument('--acType', dest='acType', type=str, default='superRes',
                    help='which model is asymetric, semSeg, or superRes')

parser.add_argument('--segRes', dest='segRes', type=str2bool, default=False, help='segnet has res skips')
parser.add_argument('--segU', dest='segU', type=str2bool, default=False, help='segnet has u skips')
parser.add_argument('--numClasses', dest='numClasses', type=str2int, default=6,
                    help='number of semantic classes for segmentation')
parser.add_argument('--use_gan', dest='use_gan', type=str2bool, default=False, help='if srgan has gan active')
# SR arguments
parser.add_argument('--nsrf', dest='nsrf', type=int, default=64, help='# of SR filters in first conv layer')
parser.add_argument('--numResBlocks', dest='numResBlocks', type=int, default=16, help='# of resBlocks in EDSR')
parser.add_argument('--sr_nc', dest='sr_nc', type=int, default=3,
                    help='# of image channels for C')  # add this for hyperspectral support
# loss coefficients
parser.add_argument('--L1_lambda', dest='L1_lambda', type=str2float, default=10.0,
                    help='weight on L1 term for normal cycle')
parser.add_argument('--idt_lambda', dest='idt_lambda', type=str2float, default=0.0,
                    help='weight assigned to the a2b identity loss function')  # b2b should give b
parser.add_argument('--tv_lambda', dest='tv_lambda', type=str2float, default=0.0,
                    help='weight assigned to the a2b total variation loss function')
parser.add_argument('--L1_sr_lambda', dest='L1_sr_lambda', type=str2float, default=10.0,
                    help='weight on L1 term in the SR cycle')  # low since patchGAN doesnt have dense summation?
parser.add_argument('--glcm_sr_lambda', dest='glcm_sr_lambda', type=str2float, default=0.0,
                    help='weight on glcm term in the SR cycle')
parser.add_argument('--idt_sr_lambda', dest='idt_sr_lambda', type=str2float, default=0.0,
                    help='weight assigned to the SR identity loss function')
parser.add_argument('--tv_sr_lambda', dest='tv_sr_lambda', type=str2float, default=0.0,
                    help='weight assigned to the SR total variation loss function')  # this is a crutch. avoid it. if needed, tune it carefuly. div2k accepts 1-2e-4, and fails at 1e-3 vs 10

# testing arguments
parser.add_argument('--testInputs', dest='testInputs', default='./datasets/grey2seg/testA',
                    help='test input images are here')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
```


 
5. For inference place validation images (png) to folder ./datasets/grey2seg/testA  

```
python main.py  --phase test --which_direction AtoB --model_dir 
```

or  testB and use --which_direction BtoA. Result from domain conversion should be in test folder.



In this example i use 12 images for train from 'CamVid data set'. You can use your images dataset or CamVid  full dataset, just download  run download_CamVid_dataset.sh

```
wget http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip
wget http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip


unzip 701_StillsRaw_full.zip
unzip LabeledApproved_full.zip

```

 convert it and  place in folders.
