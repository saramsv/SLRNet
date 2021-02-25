ls $1/ |  while read img_name
do
        img_name=$(echo $img_name | sed 's/.png//')
        id=$(echo $img_name | cut -c1-3)
        if [ -f /data/sara/TCT/CCT/body_part_sequencing/data/sequences/$id"_pcaed_sequenced" ]
        then
                corresponding_cluster_name=$(grep -w $img_name /data/sara/TCT/CCT/body_part_sequencing/data/sequences/$id"_pcaed_sequenced" | cut -d ":" -f 2)
                for cl in  $corresponding_cluster_name
                do
                        i=0
                        echo "/data/sara/semantic-segmentation-pytorch/new_train_annotations/"$img_name".png: "$cl >> data/AllSeqsWithOneAnn2beCleaned
                        if [ "$cl" != "shade" ] | [ "$cl" != "plastic" ] | [ "$cl" != "stake" ];
                        then
                                grep -w $cl$ /data/sara/TCT/CCT/body_part_sequencing/data/sequences/$id"_pcaed_sequenced" | while read img
                                do
                                echo $img >> data/AllSeqsWithOneAnn2beCleaned
                                done
                        fi
                done
        fi
done

cat data/AllSeqsWithOneAnn2beCleaned | cut -d ":" -f 2 | sort -u > data/annotated_seqnames_with_diff_sizes
cat data/annotated_seqnames_with_diff_sizes | while read line
do
	grep -w -m3 $line data/AllSeqsWithOneAnn2beCleaned >> data/annotated_seqs_only_3imgs
done
cat data/annotated_seqs_only_3imgs |sort -u > data/annotated_seqs_only_3imgs.sort
cat data/annotated_seqs_only_3imgs.sort | grep ".png" | while read line
do
        cluster_name=$(echo $line | cut -d ":" -f 2)
        img_name=$(echo $line | cut -d ":" -f 1| rev |cut -d '/' -f 1| rev | sed 's/.png//')
        echo  "/usb/seq_data_for_mit_code/"$img_name".JPG /data/sara/semantic-segmentation-pytorch/new_train_annotations/"$img_name".png" >> data/seq_data.txt
        grep -w $cluster_name data/annotated_seqs_only_3imgs.sort | grep -v ".png" | while read img
        do
                    img_name2=$(echo $img | cut -d ":" -f 1| rev |cut -d '/' -f 1| rev| sed 's/.icon//')
                    echo "/usb/seq_data_for_mit_code/"$img_name2" /data/sara/semantic-segmentation-pytorch/new_train_annotations/"$img_name".png"  >> data/seq_data.txt
        done
done
python3 data_prep.py
