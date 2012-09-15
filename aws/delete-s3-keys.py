#!/usr/bin/env python2.6

import sys
import boto
from boto.s3.key import Key

if len(sys.argv) < 3:
    print "Not enough arguments"
    print "%s <bucket> <prefix>" % sys.argv[0]
    sys.exit(1)

s3 = boto.connect_s3()
bucket = s3.get_bucket(sys.argv[1])

for k in bucket.list(sys.argv[2]):
    if k.name not in [sys.argv[2]]:
        bucket.delete_key(k.name)