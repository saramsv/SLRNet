#for i in $(seq 3 10);
#do
#	sed -i "s/list_train: .*/list_train: \"\.\/data\/mit_seq_$i\"/g" config/bodypart-10seqs-weighted-new-labels_noBack-hrnetv2.yaml
#	sed -i "s/DIR: .*/DIR: \"ckpt\/seq_$i\"/g" config/bodypart-10seqs-weighted-new-labels_noBack-hrnetv2.yaml
#	time python3 train.py --gpus 2 --cfg config/bodypart-10seqs-weighted-new-labels_noBack-hrnetv2.yaml
#done

#time CUDA_VISIBLE_DEVICES=1 python3 train.py --gpus 1 --cfg config/bodypart-mobilenetv2dilated-c1_deepsup.yaml
time CUDA_VISIBLE_DEVICES=1 python3 train.py --gpus 1 --cfg config/bodypart-resnet101-upernet.yaml
#time CUDA_VISIBLE_DEVICE=3 python3 train.py --gpus 3 --cfg config/bodypart-resnet101dilated-ppm_deepsup.yaml
#time CUDA_VISIBLE_DEVICES=2 python3 train.py --gpus 2 --cfg config/bodypart-resnet18dilated-ppm_deepsup.yaml
#time CUDA_VISIBLE_DEVICE=3 python3 train.py --gpus 3 --cfg config/bodypart-resnet50-upernet.yaml
#time CUDA_VISIBLE_DEVICES=1 python3 train.py --gpus 1 --cfg config/bodypart-resnet50dilated-ppm_deepsup.yaml
