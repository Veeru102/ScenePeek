"""S3/MinIO object storage helpers. Object keys are deterministic per video so reprocessing overwrites."""

from functools import lru_cache
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from scenepeek.core.config import get_settings


def _client(endpoint: str):
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


@lru_cache
def internal_client():
    return _client(get_settings().s3_endpoint)


@lru_cache
def public_client():
    """Client whose presigned URLs point at the browser-reachable endpoint."""
    return _client(get_settings().s3_public_endpoint)


def bucket() -> str:
    return get_settings().s3_bucket


def ensure_bucket() -> None:
    c = internal_client()
    try:
        c.head_bucket(Bucket=bucket())
    except ClientError:
        c.create_bucket(Bucket=bucket())


# ---- key layout -------------------------------------------------------------


def original_key(video_id: str, ext: str) -> str:
    return f"videos/{video_id}/original{ext}"


def web_key(video_id: str) -> str:
    return f"videos/{video_id}/web.mp4"


def poster_key(video_id: str) -> str:
    return f"videos/{video_id}/poster.jpg"


def audio_key(video_id: str) -> str:
    return f"videos/{video_id}/audio.wav"


def frame_key(video_id: str, chunk_index: int, t_ms: int) -> str:
    return f"videos/{video_id}/chunks/{chunk_index}/frames/{t_ms}.jpg"


# ---- operations -------------------------------------------------------------


def presigned_put(key: str, content_type: str, expires: int = 3600) -> str:
    return public_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket(), "Key": key, "ContentType": content_type},
        ExpiresIn=expires,
    )


def presigned_get(key: str, expires: int = 3600 * 6) -> str:
    return public_client().generate_presigned_url(
        "get_object", Params={"Bucket": bucket(), "Key": key}, ExpiresIn=expires
    )


def object_exists(key: str) -> tuple[bool, int]:
    try:
        r = internal_client().head_object(Bucket=bucket(), Key=key)
        return True, int(r.get("ContentLength", 0))
    except ClientError:
        return False, 0


def upload_file(path: Path, key: str, content_type: str | None = None) -> None:
    extra = {"ContentType": content_type} if content_type else {}
    internal_client().upload_file(str(path), bucket(), key, ExtraArgs=extra)


def upload_bytes(data: bytes, key: str, content_type: str) -> None:
    internal_client().put_object(Bucket=bucket(), Key=key, Body=data, ContentType=content_type)


def download_file(key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    internal_client().download_file(bucket(), key, str(path))


def delete_prefix(prefix: str) -> int:
    c = internal_client()
    paginator = c.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            c.delete_objects(Bucket=bucket(), Delete={"Objects": objs})
            deleted += len(objs)
    return deleted
