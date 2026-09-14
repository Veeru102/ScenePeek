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
    queues: str = typer.Option(None, help="Comma-separated queues, e.g. cpu,ml,vision"),
    once: bool = typer.Option(False, help="Process one job and exit"),
    min_priority: int = typer.Option(None, help="Refuse jobs below this priority (0 = interactive only)"),
):
    """Run a background worker that leases and executes jobs."""
    from scenepeek.jobs.worker import run_worker

    run_worker(queues=queues.split(",") if queues else None, once=once, min_priority=min_priority)


@app.command("reindex-captions")
def reindex_captions(
    video_id: str = typer.Option(None, help="Only this video (default: all)"),
    force: bool = typer.Option(False, help="Re-caption segments that already have captions"),
):
    """Backfill keyframe captions for already-indexed videos without a full reindex."""
    from scenepeek.pipeline.captions import backfill

    backfill(video_id=video_id, force=force)


@app.command("backfill")
def backfill_cmd(
    kind: str = typer.Argument(help="temporal"),
    model: str = typer.Option(None, help="Encoder to backfill (default: the configured one)"),
    video_id: str = typer.Option(None, help="Only this video"),
    force: bool = typer.Option(False, help="Re-encode chunks that already have rows for this model"),
):
    """Enqueue low-priority jobs that add a model version's representations next to the existing ones."""
    from sqlalchemy import create_engine

    from scenepeek.core.config import get_settings
    from scenepeek.pipeline import temporal

    if kind != "temporal":
        raise typer.BadParameter("only 'temporal' can be backfilled right now")
    n = temporal.backfill(
        create_engine(get_settings().sync_database_url), model=model, video_id=video_id, force=force
    )
    typer.echo(f"enqueued {n} encode_temporal jobs at backfill priority")


@app.command("versions")
def versions_cmd(retire: str = typer.Option(None, help="kind:model_key to retire")):
    """List model/index versions (what the stored representations were produced by)."""
    from sqlalchemy import create_engine

    from scenepeek.core.config import get_settings
    from scenepeek.pipeline import temporal

    engine = create_engine(get_settings().sync_database_url)
    if retire:
        kind, key = retire.split(":", 1)
        temporal.retire(engine, key)
    for v in temporal.versions(engine):
        typer.echo(f"{v.kind:<9} {v.model_key:<40} dim={v.dim or '-':<5} {v.status:<9} {v.index_name or ''}")


@app.command("mine-negatives")
def mine_negatives(
    experiment: str = typer.Argument(help="Experiment name or id (its recorded hits are mined)"),
    max_iou: float = typer.Option(0.1, help="A hit overlapping the truth more than this is not a negative"),
    per_query: int = typer.Option(5),
    out: str = typer.Option(None, help="JSONL export path (default data/negatives/<experiment>.jsonl)"),
):
    """Persist (query, positive, hard negative) triples from retrieval failures for reranker training."""
    from pathlib import Path

    from scenepeek.eval.negatives import mine

    path = Path(out) if out else Path("../data/negatives") / f"{experiment}.jsonl"
    m = mine(experiment, max_iou=max_iou, per_query=per_query, out=path)
    typer.echo(
        f"{m.queries} queries -> {m.positives} positives, {m.negatives} hard negatives "
        f"({m.skipped_no_positive} skipped: no indexed segment overlaps the truth); wrote {path}"
    )


router_app = typer.Typer(help="Learned query routing")
app.add_typer(router_app, name="router")


@router_app.command("train")
def router_train(
    experiment: str = typer.Argument(
        help="Experiment name or id whose lane attribution is the training signal"
    ),
    version: str = typer.Option(
        "v1", help="Saved as models/router/<version>.joblib; use router: learned:<version>"
    ),
    holdout: float = typer.Option(0.2),
    seed: int = typer.Option(0),
    min_p: float = typer.Option(0.15, help="Lanes predicted below this probability are skipped"),
):
    """Fit per-lane usefulness classifiers from a recorded experiment (labels = did the lane's own
    top-10 contain the answer) and report holdout AUC per lane."""
    from scenepeek.search.learned_router import train

    m = train(experiment, version, holdout=holdout, seed=seed, min_p=min_p)
    typer.echo(
        f"router {version}: lanes={m.lanes} n_train={m.meta['n_train']} n_holdout={m.meta['n_holdout']}"
    )
    for lane in m.lanes:
        auc = m.meta["holdout_auc"].get(lane)
        typer.echo(f"  {lane:<9} prior={m.priors[lane]:.2f}  holdout AUC={auc if auc is not None else 'n/a'}")


eval_app = typer.Typer(help="Search quality evaluation")
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    config: str = typer.Option("../eval/experiments/synthetic.yaml", "-c"),
    output: str = typer.Option(None, "-o"),
):
    """Run an experiment YAML (eval/experiments/*.yaml). Legacy eval/configs files still work."""
    import yaml

    from scenepeek.eval.experiment import ExperimentSpec, run
    from scenepeek.eval.runner import run_eval

    data = yaml.safe_load(open(config)) or {}
    if str(data.get("dataset", "")).endswith(".yaml") or "overrides" in data:
        run_eval(config, output)
    else:
        run(ExperimentSpec.from_yaml(config), output)


@eval_app.command("list")
def eval_list(limit: int = typer.Option(20)):
    """Recent experiments recorded in Postgres."""
    from sqlalchemy import select

    from scenepeek.models import Experiment

    with _sync_session() as s:
        rows = s.scalars(select(Experiment).order_by(Experiment.started_at.desc()).limit(limit))
        for e in rows:
            m = e.metrics or {}
            head = ", ".join(f"{k}={m[k]:.3f}" for k in ("mrr", "r1@0.5", "map") if k in m)
            when = f"{e.started_at:%Y-%m-%d %H:%M}"
            typer.echo(f"{when}  {str(e.id)[:8]}  {e.status:<7} {e.name:<32} n={e.n_queries:<4} {head}")


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
