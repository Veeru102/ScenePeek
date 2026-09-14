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


if __name__ == "__main__":
    app()
