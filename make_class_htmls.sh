cat /data/sara/TCT/CCT/body_part_sequencing/data/expBeforeFeb24th21Data/bodyPartAnntags.csv.20210220FixedLabels | grep $1 > $1"_tags"
cat $1"_tags" | awk -F "," '{print $(NF -4)}' | rev |cut -d "/" -f 1| rev | sed 's/.JPG/.icon.JPG/g' | sort -u > $1"_imgs" 
rm $1"_imgs_for_da1"
cat $1"_imgs" | while read line
do
	id=$(echo $line| cut -c1-3)
	echo "/home/mousavi/da1/icputrd/arf/mean.js/public/sara_img/"$id"/"$line":"$1 >> $1"_imgs_for_da1"
done
cp $1"_imgs_for_da1"  /da1_data/icputrd/visulizeClusters/static/Cluster-HTMLs/ClusterAll1Million/

