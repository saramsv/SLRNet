
python3 evaluate.py --cfg $1
#python3 evaluate.py --cfg config/bodypart_eval_balanced_seqs_7class_hrnetv2.yaml
#for i in $(seq 2 2);
#do
#	echo seq$i
#	sed -i "s/list_train: .*/list_train: \"\.\/data\/mit_seq_$i\"/g" config/bodypart-10seqs-weighted-new-labels_noBack-hrnetv2.yaml
#	sed -i "s/DIR: .*/DIR: \"ckpt\/seq_$i\"/g" config/bodypart-10seqs-weighted-new-labels_noBack-hrnetv2.yaml
#	sed -i "s/checkpoint: .*/checkpoint: \"epoch_best.pth\"/g" config/bodypart-10seqs-weighted-new-labels_noBack-hrnetv2.yaml
#	python3 evaluate.py --cfg config/bodypart-10seqs-weighted-new-labels_noBack-hrnetv2.yaml
#done
#python3 train.py --gpus 3 --cfg config/bodypart-mobilenetv2dilated-c1_deepsup.yaml
#python3 train.py --gpus 3 --cfg config/bodypart-resnet101-upernet.yaml
#python3 train.py --gpus 3 --cfg config/bodypart-resnet101dilated-ppm_deepsup.yaml
#python3 train.py --gpus 3 --cfg config/bodypart-resnet18dilated-ppm_deepsup.yaml
#python3 train.py --gpus 3 --cfg config/bodypart-resnet50-upernet.yaml
#python3 train.py --gpus 3 --cfg config/bodypart-resnet50dilated-ppm_deepsup.yaml
