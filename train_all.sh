# time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-hrnetv2.yaml
# time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-mobilenetv2dilated-c1_deepsup.yaml
# time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-resnet101-upernet.yaml
# time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-resnet101dilated-ppm_deepsup.yaml
# time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-resnet18dilated-ppm_deepsup.yaml
# time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-resnet50-upernet.yaml
# time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-resnet50dilated-ppm_deepsup.yaml

#{ time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-hrnetv2.yaml ; } > tmp_results/bodypart-hrnetv2.res 2>&1 

#{ time CUDA_VISIBLE_DEVICES=2 python3 train.py --gpus 2 --cfg config/bodypart-resnet50-upernet.yaml ; } > tmp_results/bodypart-resnet50-upernet.res 2>&1 
#{ bash eval.sh config/bodypart-resnet50-upernet.yaml ; } >> tmp_results/bodypart-resnet50-upernet.res 2>&1 

#{ time CUDA_VISIBLE_DEVICES=2 python3 train.py --gpus 2 --cfg config/bodypart-resnet50dilated-ppm_deepsup.yaml ; } > tmp_results/bodypart-resnet50dilated-ppm_deepsup.res 2>&1 
#{ bash eval.sh  config/bodypart-resnet50dilated-ppm_deepsup.yaml ; } >> tmp_results/bodypart-resnet50dilated-ppm_deepsup.res 2>&1 

#{ time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-mobilenetv2dilated-c1_deepsup.yaml ; } > tmp_results/bodypart-mobilenetv2dilated-c1_deepsup.res 2>&1 
#{ bash eval.sh config/bodypart-mobilenetv2dilated-c1_deepsup.yaml ; } >> tmp_results/bodypart-mobilenetv2dilated-c1_deepsup.res 2>&1 

#{ time CUDA_VISIBLE_DEVICES=1 python3 train.py --gpus 1 --cfg config/bodypart-resnet101-upernet_noweight.yaml ; } > tmp_results/bodypart-resnet101-upernet_noweight.res 2>&1 
#{ bash eval.sh config/bodypart-resnet101-upernet_noweight.yaml  ; } >> tmp_results/bodypart-resnet101-upernet_noweight.res 2>&1 

#{ time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-resnet101dilated-ppm_deepsupV2.yaml ; } > tmp_results/bodypart-resnet101dilated-ppm_deepsup.res 2>&1 
#{ bash eval.sh config/bodypart-resnet101dilated-ppm_deepsupV2.yaml ; } >> tmp_results/bodypart-resnet101dilated-ppm_deepsup.res 2>&1 

#{ time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 3 --cfg config/bodypart-resnet18dilated-ppm_deepsup.yaml ; } > tmp_results/bodypart-resnet18dilated-ppm_deepsup.res 2>&1 
#{ bash eval.sh  config/bodypart-resnet18dilated-ppm_deepsup.yaml ; } >> tmp_results/bodypart-resnet18dilated-ppm_deepsup.res 2>&1 


# random resnet and xception
{ time CUDA_VISIBLE_DEVICES=2 python3 train.py --gpus 1 --cfg config/random_resnet101_upernet.yaml ; } > tmp_results/random_resnet101_upernet.res 2>&1 
{ bash eval.sh config/random_resnet101_upernet.yaml  ; } >> tmp_results/random_resnet101_upernet.res 2>&1 


#{ time CUDA_VISIBLE_DEVICES=3 python3 train.py --gpus 1 --cfg config/random_xception_c1.yaml ; } > tmp_results/random_xception_c1.res 2>&1 
#{ bash eval.sh config/random_xception_c1.yaml ; } >> tmp_results/random_xception_c1.res 2>&1 