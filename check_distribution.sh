cat $1 | cut -d ":" -f 2| cut -d "," -f 1 | rev | cut -d "/" -f 1 | rev | cut -d "\"" -f 1 |sort -u | while read line
#ls $1 | while read line
do 
	#line=$(echo $line | sed 's/png/JPG/g')
	grep $line data/sequence_data/bodyPartAnntags.csv.20210220FixedLabels >> labels
done

cat labels | awk -F "," '{print $(NF -3)}' | sort | uniq -c
rm labels
