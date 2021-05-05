#files=`ls /data/sara/semantic-segmentation-pytorch/data/test_imgs/*JPG`
#files=`ls $1/*JPG`
files=`ls $1/*png`
for f in $files
do
	python3 predict_img.py -i $f --cfg $2 --save $3
	#python3 predict_img.py -i $f --cfg "config/bodypart_eval_balanced_seqs_7class_hrnetv2.yaml"
done
