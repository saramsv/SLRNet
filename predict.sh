files=`ls data/*JPG`
for f in $files
do
	python3 predict_img.py -i $f --cfg "config/bodypart-notseq-hrnetv2.yaml"
done
