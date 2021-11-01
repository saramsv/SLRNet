#bash eval_window.sh 243 /data/sara/semantic-segmentation-pytorch/data/for_pair_iou_test2/243_imgs /data/sara/semantic-segmentation-pytorch/data/for_pair_iou_test2/ > window_eval_res 2>&1
for w in $( seq 2 10);
do
	echo "################# INFO: window size is: "$w"###########"
	donor_id=$1 #243
	labeled_imgs=$2 #/data/sara/semantic-segmentation-pytorch/data/for_pair_iou_test/labeled_imgs_for_donor_243
	dest_dir=$3 #/data/sara/semantic-segmentation-pytorch/data/for_pair_iou_test2/
	cd /data/sara/TCT/CCT/body_part_sequencing/scripts/
	time python3 decom_sequence_generator_keras_pcaed.py --path $2 --donor_id $donor_id --dest_dir $3 --window $w

	cd /data/sara/semantic-segmentation-pytorch/
	#bash get_seq_with_ann.sh all_body_part_annotations/ data/for_pair_iou_test2/ data/for_pair_iou_test2/matches sequencedpca
	#sed -i 's/ //g' data/for_pair_iou_test2/matches
	#cat data/for_pair_iou_test2/matches | grep -v png| sort -u > data/for_pair_iou_test2/matches2
	#rm data/for_pair_iou_test2/matches
	#mv data/for_pair_iou_test2/matches2 data/for_pair_iou_test2/matches

	#python3 create_pairs_from_labeled_clusters.py  data/for_pair_iou_test2/matches data/for_pair_iou_test2/matches

	python3 eval_purity.py  $3$1"sequencedpca"$w $3$1"sequencedpca"$w
done
