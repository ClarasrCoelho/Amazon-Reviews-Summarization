# serving_api/s3_utils.py
import os
import json
import boto3
from botocore.exceptions import ClientError

AWS_PROFILE = os.getenv("AWS_PROFILE", "")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")

def _session():
    if AWS_PROFILE:
        return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return boto3.Session(region_name=AWS_REGION)

def s3_client():
    return _session().client("s3")

def read_text_from_s3(bucket: str, key: str) -> str:
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8", errors="replace")

def write_text_to_s3(bucket: str, key: str, text: str, content_type: str = "text/plain"):
    s3 = s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)

def read_json_from_s3(bucket: str, key: str) -> dict | None:
    try:
        txt = read_text_from_s3(bucket, key)
        return json.loads(txt)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    except Exception:
        return None

def write_json_to_s3(bucket: str, key: str, obj: dict):
    write_text_to_s3(bucket, key, json.dumps(obj, ensure_ascii=False), content_type="application/json")

def list_keys(bucket: str, prefix: str):
    s3 = s3_client()
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            keys.append(it["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

def append_jsonl(bucket: str, key: str, record: dict):
    s3 = s3_client()
    try:
        current = read_text_from_s3(bucket, key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            current = ""
        else:
            raise

    line = json.dumps(record, ensure_ascii=False)
    new_body = (current + ("\n" if current and not current.endswith("\n") else "") + line + "\n")
    write_text_to_s3(bucket, key, new_body, content_type="application/jsonl")
