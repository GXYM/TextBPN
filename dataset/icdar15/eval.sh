cd dataset/icdar15/
rm submit/*
cp $1/*.txt submit
cd submit/;zip -r  submit.zip * &> ../log.txt ;mv submit.zip ../; cd ../
rm log.txt
python Evaluation_Protocol/script.py -g=gt.zip -s=submit.zip
