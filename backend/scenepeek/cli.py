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


if __name__ == "__main__":
    app()
