"""Upload eval/real/sources.yaml videos into the running ScenePeek library via the API."""

import mimetypes

import httpx

from scenepeek.eval.real import REPO_ROOT, SOURCES_PATH, load_sources

API_BASE = "http://localhost:8000/api"


def upload_all(api_base: str = API_BASE) -> None:
    sources = load_sources(SOURCES_PATH)

    with httpx.Client(timeout=60) as client:
        existing_titles = {v["title"] for v in client.get(f"{api_base}/videos").json()}

        for v in sources:
            if v["title"] in existing_titles:
                print(f"skip (already in library): {v['title']}")
                continue

            path = REPO_ROOT / v["file"]
            if not path.exists():
                print(f"MISSING file, skipping: {path}")
                continue

            content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
            size_bytes = path.stat().st_size

            create_res = client.post(
                f"{api_base}/videos",
                json={
                    "filename": path.name,
                    "title": v["title"],
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                },
            )
            create_res.raise_for_status()
            target = create_res.json()

            print(f"uploading {v['title']} ({size_bytes / 1e6:.0f} MB)...")
            with path.open("rb") as f:
                put_res = client.put(
                    target["upload_url"],
                    content=f.read(),
                    headers={"Content-Type": content_type},
                    timeout=600,
                )
            put_res.raise_for_status()

            complete_res = client.post(f"{api_base}/videos/{target['video_id']}/complete")
            complete_res.raise_for_status()
            print(f"  done -> queued for indexing (video_id={target['video_id']})")


if __name__ == "__main__":
    upload_all()
