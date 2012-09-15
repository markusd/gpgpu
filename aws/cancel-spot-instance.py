#!/usr/bin/env python2.6

import sys
import time
from boto import ec2

if __name__== '__main__':
    con = ec2.connect_to_region("us-east-1")

    spot_requests = con.get_all_spot_instance_requests(str(sys.argv[1]))

    for request in spot_requests:
        print "Found request %s" % request
        if request.instance_id is None:
            print "Unable to get instance id of request %s" % request
            request.cancel()
            continue

        instances = con.get_all_instances(str(request.instance_id))

        for instance in instances:
            for running_instance in instance.instances:
                print "Terminate instance %s" % running_instance
                running_instance.terminate()

        request.cancel()