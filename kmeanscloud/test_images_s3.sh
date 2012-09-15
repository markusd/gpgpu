#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Not enough arguments"
    exit 1
fi

HOST=$1
IMAGE_COUNT=$2

S3_BUCKET=gpu_images
S3_FOLDER=output
INPUT_FOLDER=images
OUTPUT_FOLDER=output
UPLOADER="" #"\"python upload.py $S3_BUCKET $S3_FOLDER\""

rm kmeanscloud.tgz
tar cfvz kmeanscloud.tgz kmeanscloud/

scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa images.txt kmeanscloud.tgz .boto ec2-user@$HOST:~/

ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa ec2-user@$HOST "
    rm -rf kmeanscloud &&
    tar xfzv kmeanscloud.tgz &&
    mkdir -p kmeanscloud/Release/$INPUT_FOLDER &&
    mkdir -p kmeanscloud/Release/$OUTPUT_FOLDER &&
    mv images.txt kmeanscloud/Release/ &&
    cd kmeanscloud/Release &&
    make all &&
    cd $INPUT_FOLDER &&
    time \`head -n $IMAGE_COUNT ../images.txt | xargs -n 1 -P 100 wget\` ;
    cd .. &&
    time ./start.sh $INPUT_FOLDER $OUTPUT_FOLDER $UPLOADER < input ;
    time python upload.py $S3_BUCKET $S3_FOLDER $OUTPUT_FOLDER"
