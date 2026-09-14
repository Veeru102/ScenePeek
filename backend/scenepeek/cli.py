import typer

from scenepeek.core.logging import configure_logging

app = typer.Typer(help="ScenePeek command line", no_args_is_help=True)


@app.callback()
def _root(log_level: str = typer.Option("INFO", envvar="LOG_LEVEL")):
    configure_logging(log_level)


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("scenepeek.api.app:app", host=host, port=port, reload=reload)


@app.command()
def migrate():
    """Apply database migrations."""
    from pathlib import Path

    from alembic.config import Config

    from alembic import command

    cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    command.upgrade(cfg, "head")


@app.command()
def worker(
    queues: str = typer.Option(None, help="Comma-separated queues, e.g. cpu,ml"),
    once: bool = typer.Option(False, help="Process one job and exit"),
):
    """Run a background worker that leases and executes jobs."""
    from scenepeek.jobs.worker import run_worker

    run_worker(queues=queues.split(",") if queues else None, once=once)


@app.command("reindex-captions")
def reindex_captions(
    video_id: str = typer.Option(None, help="Only this video (default: all)"),
    force: bool = typer.Option(False, help="Re-caption segments that already have captions"),
):
    """Backfill keyframe captions for already-indexed videos without a full reindex."""
    from scenepeek.pipeline.captions import backfill

    backfill(video_id=video_id, force=force)


eval_app = typer.Typer(help="Search quality evaluation")
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    config: str = typer.Option("../eval/configs/default.yaml", "-c"),
    output: str = typer.Option(None, "-o"),
):
    from scenepeek.eval.runner import run_eval

    run_eval(config, output)


@eval_app.command("compare")
def eval_compare(reports: list[str]):
    from scenepeek.eval.report import compare_reports

    compare_reports(reports)


@eval_app.command("real-scan")
def eval_real_scan(
    videos: str = typer.Option(None, help="Comma-separated video keys to (re)scan; default: all unscanned"),
    force: bool = typer.Option(False, help="Regenerate candidates for the given videos, dropping old ones"),
):
    """Analyze eval/real/sources.yaml videos and propose candidate queries."""
    from scenepeek.eval.real import scan_videos

    scan_videos(video_keys=videos.split(",") if videos else None, force=force)


@eval_app.command("real-review")
def eval_real_review(type: str = typer.Option(None, "--type", help="Only review this candidate type")):
    """Interactively approve/reject/edit candidates into eval/real/dataset.yaml."""
    from scenepeek.eval.review import review

    review(type_filter=type)


@eval_app.command("real-review-web")
def eval_real_review_web(port: int = typer.Option(5173, help="Port to run web UI on")):
    """Launch web UI for reviewing candidates (open http://localhost:PORT)."""
    import uvicorn

    from scenepeek.eval.web_review import app

    print(f"\n🌐 Opening http://localhost:{port} in your browser...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


@eval_app.command("real-label")
def eval_real_label(port: int = typer.Option(5175, help="Port for the labeling UI")):
    """Hand-write real search queries against eval/real videos -> eval/real/human.yaml."""
    import socket

    import uvicorn

    from scenepeek.eval.web_label import HUMAN_PATH, app

    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise typer.BadParameter(f"port {port} is already in use — is another labeling server running?")
    print(f"\nLabeling UI: http://localhost:{port}  (writes {HUMAN_PATH}; Ctrl+C when done)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


@eval_app.command("real-auto-review")
def eval_real_auto_review():
    """Automatically review all candidates using heuristics."""
    from scenepeek.eval.auto_review import auto_review

    auto_review()


@eval_app.command("real-upload")
def eval_real_upload():
    """Upload eval/real/sources.yaml videos into the running library for indexing."""
    from scenepeek.eval.upload_real import upload_all

    upload_all()


dataset_app = typer.Typer(help="Benchmark datasets (QVHighlights, the ScenePeek YAML sets)")
app.add_typer(dataset_app, name="dataset")


def _sync_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from scenepeek.core.config import get_settings

    return Session(create_engine(get_settings().sync_database_url), expire_on_commit=False)


@dataset_app.command("import")
def dataset_import(
    name: str = typer.Argument(help="qvhighlights | scenepeek_human | scenepeek_auto | scenepeek_synthetic"),
    dir: str = typer.Option("..", "--dir", help="Directory holding the annotation files"),
    split: str = typer.Option("val"),
    limit: int = typer.Option(None, help="Seeded random subset of videos (all their queries come along)"),
    seed: int = typer.Option(0),
):
    """Register a dataset's videos + queries in Postgres (idempotent). Fetching is a separate step."""
    from pathlib import Path

    from scenepeek.datasets.importer import import_split
    from scenepeek.datasets.registry import get_adapter

    adapter = get_adapter(name)
    with _sync_session() as s:
        r = import_split(s, adapter, Path(dir), split, limit=limit, seed=seed)
    typer.echo(f"{name}/{split}: {r.videos} videos, {r.queries} queries, {r.linked} already in the library")


@dataset_app.command("fetch")
def dataset_fetch(
    name: str = typer.Argument(),
    split: str = typer.Option(None),
    limit: int = typer.Option(None, help="Enqueue at most this many fetches"),
):
    """Enqueue fetch jobs (benchmark priority) for every pending video; workers do the rest."""
    from scenepeek.datasets.importer import enqueue_fetches, get_or_create_dataset
    from scenepeek.datasets.registry import get_adapter

    with _sync_session() as s:
        ds = get_or_create_dataset(s, get_adapter(name))
        n = enqueue_fetches(s, ds, split=split, limit=limit)
    typer.echo(f"enqueued {n} fetch jobs")


@dataset_app.command("status")
def dataset_status(name: str = typer.Argument()):
    """Per-split counts of pending / fetched / indexed / unavailable videos and queries."""
    from scenepeek.datasets.importer import get_or_create_dataset, link_library_videos, status
    from scenepeek.datasets.registry import get_adapter

    with _sync_session() as s:
        ds = get_or_create_dataset(s, get_adapter(name))
        link_library_videos(s, ds)
        s.commit()
        for split, counts in status(s, ds).items():
            typer.echo(f"{split}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    app()
