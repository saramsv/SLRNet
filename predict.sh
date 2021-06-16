#bash predict.sh unlabeled_imgs_in_pairs.sort config/bodypart_hrnet_sup.yaml /usb/hrnetSup_teacher_pseudo_labels/
#files=`ls /data/sara/semantic-segmentation-pytorch/data/test_imgs/*JPG`
#files=`ls $1/*JPG`
##files=`ls $1/*png`
##for f in $files
while read -r f 
do
	python3 predict_img.py -i $f --cfg $2 --save $3
	#python3 predict_img.py -i $f --cfg "config/bodypart_eval_balanced_seqs_7class_hrnetv2.yaml"
done < $1
