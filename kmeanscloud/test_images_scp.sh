#!/bin/bash

# SOURCE_FOLDER NEEDS TO HAVE THE RIGHT AMOUNT OF IMAGES

if [ "$#" -lt 3 ]; then
    echo "Not enough arguments"
    exit 1
fi

HOST=$1
SOURCE_FOLDER=$2
DESTINATION_FOLDER=$3

S3_BUCKET=gpu_images
S3_FOLDER=output
INPUT_FOLDER=images
OUTPUT_FOLDER=output
UPLOADER="" #"\"python upload.py $S3_BUCKET $S3_FOLDER\""

rm kmeanscloud.tgz
tar cfvz kmeanscloud.tgz kmeanscloud/

scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa images.txt kmeanscloud.tgz .boto ec2-user@$HOST:~/

time ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa ec2-user@$HOST "
    rm -rf kmeanscloud &&
    tar xfzv kmeanscloud.tgz &&
    mkdir -p kmeanscloud/Release/$INPUT_FOLDER &&
    mkdir -p kmeanscloud/Release/$OUTPUT_FOLDER &&
    mv images.txt kmeanscloud/Release/ &&
    cd kmeanscloud/Release &&
    make all"

time scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa $SOURCE_FOLDER/* ec2-user@$HOST:~/kmeanscloud/Release/$INPUT_FOLDER/

time ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa ec2-user@$HOST "
    cd kmeanscloud/Release &&
    ./start.sh $INPUT_FOLDER $OUTPUT_FOLDER $UPLOADER < input"

time scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa ec2-user@$HOST:~/kmeanscloud/Release/$OUTPUT_FOLDER/* $DESTINATION_FOLDER
