#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import boto3
import sys

sys.path.append('./site-packages')

from crhelper import CfnResource

helper = CfnResource()

def delete_s3(event):
    s3_resource = boto3.resource("s3")

    bucket_name = event["ResourceProperties"]["S3BucketName"]
    try:
        s3_resource.Bucket(bucket_name).objects.all().delete()
        print("Successfully deleted objects in bucket "
                "called '{}'".format(bucket_name))
            
    except s3_resource.meta.client.exceptions.NoSuchBucket:
        print(
            "Could not find bucket called '{}'. "
            "Skipping delete.".format(bucket_name)
        )

@helper.update
@helper.create
def empty_function(event, _):
    pass

@helper.delete
def on_delete(event, _):
    delete_s3(event)
    
def handler(event, context):
    helper(event, context)
