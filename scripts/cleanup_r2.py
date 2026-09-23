"""Delete only cloth-agent relay PNGs older than an hour; scheduled externally."""
import datetime
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.r2_config import load_r2_env
import boto3
from botocore.config import Config


def cleanup(client, bucket, now):
    cutoff = now - datetime.timedelta(hours=1)
    count = 0
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix='cloth-agent/'):
        keys = [{'Key': obj['Key']} for obj in page.get('Contents', [])
                if re.fullmatch(r'cloth-agent/[0-9a-f]{32}\.png', obj['Key'])
                and obj['LastModified'] <= cutoff]
        if keys:
            result = client.delete_objects(Bucket=bucket, Delete={'Objects': keys, 'Quiet': True})
            if result.get('Errors'):
                raise RuntimeError('R2 cleanup returned deletion errors')
            count += len(keys)
    return count


if __name__ == '__main__':
    load_r2_env()
    client = boto3.client('s3',
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], region_name='auto',
        config=Config(signature_version='s3v4', connect_timeout=10, read_timeout=30,
                      retries={'mode': 'standard', 'total_max_attempts': 2}))
    print('Deleted relay objects:', cleanup(client, os.environ['R2_BUCKET'], datetime.datetime.now(datetime.timezone.utc)))
