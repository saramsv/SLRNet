#!/bin/bash 

#cat $1 | while read line
#do
#    grep $line $2
#done
cat decom_notseq_data.odgt | while read line;
do 
	name=$(echo $line|cut -d "," -f1| cut -d "G" -f 1| cut -d "u" -f 2) 
	grep  $name clean_decom_seq_dataV17.odgt #decom_seq_data.odgt
done
