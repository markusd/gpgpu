#!/usr/bin/env python2.6

import sys
import time
from boto import ec2

if __name__== '__main__':
    con = ec2.connect_to_region("us-east-1")

    # Create a single request
    spot_requests = con.request_spot_instances(price="0.50", count=1, image_id="ami-47872e2e", key_name="awskey", security_groups=["default"], instance_type="cg1.4xlarge")

    if len(spot_requests) != 1:
        print "Failed to create spot request"
        sys.exit(1)

    # Get the request id
    request = spot_requests[0]
    print "Created request %s" % request

    tries = 0

    while request.instance_id is None and tries < 180:
        tries += 1
        time.sleep(1)
        spot_requests = con.get_all_spot_instance_requests(str(request.id))

        if len(spot_requests) != 1:
            print "Failed to get spot requests, cancelling"
            request.cancel()
            sys.exit(1)

        request = spot_requests[0]

    if request.instance_id is None:
        print "Unable to get instance id of request, cancelling..."
        request.cancel()
        sys.exit(1)

    instances = con.get_all_instances(str(request.instance_id))

    if len(instances) < 1:
        print "Failed to get instance %s, cancelling" % request.instance_id
        request.cancel()
        sys.exit(1)

    instance = instances[0]
    print "Created instance %s" % instance

    tries = 0

    while len(instance.instances) < 1 or instance.instances[0].public_dns_name in [None, "", u""]:
        tries += 1
        time.sleep(1)

        instances = con.get_all_instances(str(request.instance_id))

        if len(instances) < 1:
            print "Failed to get instance %s, cancelling" % request.instance_id
            request.cancel()
            sys.exit(1)

        instance = instances[0]

    running_instance = instance.instances[0]

    print "Host:%s" % running_instance.public_dns_name
    