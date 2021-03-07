files=`ls /data/sara/semantic-segmentation-pytorch/data/img_examples/*JPG`
for f in $files
do
	python3 predict_img.py -i $f --cfg $1
	#python3 predict_img.py -i $f --cfg "config/bodypart_eval_balanced_seqs_7class_hrnetv2.yaml"
done
