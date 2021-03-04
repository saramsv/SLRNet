files=`ls /data/sara/semantic-segmentation-pytorch/data/img_examples/*JPG`
for f in $files
do
	python3 predict_img.py -i $f --cfg "config/bodypart_eval_balancedall_seq_with_one_ann_weighted_hrnetv2.yaml"
done
