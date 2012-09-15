#!/bin/bash

# SOURCE_FOLDER NEEDS TO HAVE THE RIGHT AMOUNT OF IMAGES

if [ "$#" -lt 2 ]; then
    echo "Not enough arguments"
    exit 1
fi

HOST=$1
DESTINATION_FOLDER=$2


rm kmeanscloud2.tgz
tar cfvz kmeanscloud2.tgz kmeanscloud2/

scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa kmeanscloud2.tgz ec2-user@$HOST:~/

time ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa ec2-user@$HOST "
    rm -rf kmeanscloud2 &&
    tar xfzv kmeanscloud2.tgz &&
    cd kmeanscloud2/Release &&
    time make all &&
    time ./start.sh < input"

time scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/awskey_rsa ec2-user@$HOST:~/kmeanscloud/Release/*.out $DESTINATION_FOLDER
