cat /data/sara/TCT/CCT/body_part_sequencing/data/expBeforeFeb24th21Data/bodyPartAnntags.csv.20210220FixedLabels | grep $1 > $1"_tags"
cat $1"_tags" | awk -F "," '{print $(NF -4)}' | rev |cut -d "/" -f 1| rev | sed 's/.JPG//g' | sort -u > $1"_imgs" 
bash grep.sh $1"_imgs" /data/sara/TCT/CCT/body_part_sequencing/data/10SeqsForCCT/cct_seq_10 > $1"_seqs" 
sed -i 's/ /\n/g' $1"_seqs"
sed -i 's/ //g' $1"_seqs"

rm $1"_seqs_for_da1"

cat $1"_seqs" | while read line
do
	img_name=$(echo $line| rev|cut -d '/' -f 1| rev|sed 's/JPG/icon.JPG/g')
	id=$(echo $img_name| cut -c1-3)
	echo "/home/mousavi/da1/icputrd/arf/mean.js/public/sara_img/"$id"/"$img_name":"$1 >> $1"_seqs_for_da1"
done
cp $1"_seqs_for_da1"  /da1_data/icputrd/visulizeClusters/static/Cluster-HTMLs/ClusterAll1Million/

