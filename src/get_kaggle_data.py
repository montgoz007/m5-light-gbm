import os
import subprocess
import zipfile
import shutil
from dotenv import load_dotenv
import typer
import pandas as pd
from sklearn.model_selection import train_test_split

app = typer.Typer(help="Fetch, process & split a Kaggle competition dataset.")

load_dotenv()  # loads KAGGLE_USERNAME & KAGGLE_KEY

def human_size(n_bytes: int) -> str:
    for unit in ('B','KiB','MiB','GiB','TiB'):
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}PiB"

def dir_size(path: str) -> str:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return human_size(total)

@app.command()
def main(
    competition: str = typer.Argument(..., help="Kaggle competition slug (e.g. titanic)"),
    test_size: float = typer.Option(0.2, help="Fraction to hold out if no test set provided")
):
    raw_dir    = os.path.join("data", competition, "raw")
    proc_dir   = os.path.join("data", competition, "processed")
    splits_dir = os.path.join(proc_dir, "splits")

    # 1) DOWNLOAD & UNZIP (idempotent)
    os.makedirs(raw_dir, exist_ok=True)
    csvs = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if csvs:
        typer.secho("⚠️  CSVs already in place — skipping download.", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"⏬ Downloading {competition}", fg=typer.colors.CYAN)
        cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", raw_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            typer.secho("❌ Download failed", fg=typer.colors.RED)
            typer.echo(res.stderr or res.stdout)
            raise typer.Exit(1)
        for z in os.listdir(raw_dir):
            if z.endswith(".zip"):
                typer.secho(f"📂 Unzipping {z}", fg=typer.colors.CYAN)
                with zipfile.ZipFile(os.path.join(raw_dir, z)) as zf:
                    zf.extractall(raw_dir)
                os.remove(os.path.join(raw_dir, z))
        csvs = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

    # 2) CONVERT CSV → PARQUET & ANONYMIZE
    os.makedirs(proc_dir, exist_ok=True)
    parquets = []
    for csv in csvs:
        src = os.path.join(raw_dir, csv)
        dst = os.path.join(proc_dir, csv.replace(".csv", ".parquet"))
        if not os.path.exists(dst):
            typer.secho(f"🔄 Converting {csv}", fg=typer.colors.CYAN)
            df = pd.read_csv(src)
            # drop common PII columns
            for col in ("Name","Ticket","Cabin"):
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
            df.to_parquet(dst, index=False)
        parquets.append(os.path.basename(dst))

    # 3) REPORT DIR SIZES & DATA SHAPES
    typer.secho("\nℹ️  SUMMARY:", fg=typer.colors.GREEN)
    typer.echo(f" Raw: {raw_dir} ({dir_size(raw_dir)})")
    typer.echo(f" Proc: {proc_dir} ({dir_size(proc_dir)})\n")
    for pq in parquets:
        df = pd.read_parquet(os.path.join(proc_dir, pq))
        typer.echo(f" • {pq}: shape={df.shape}, types={df.dtypes.to_dict()}")

    # 4) SPLIT HANDLING
    os.makedirs(splits_dir, exist_ok=True)
    # detect existing split files
    has_train = "train.parquet" in parquets
    has_test  = "test.parquet"  in parquets

    out_train = os.path.join(splits_dir, "train.parquet")
    out_test  = os.path.join(splits_dir, "test.parquet")

    if os.path.exists(out_train) and os.path.exists(out_test):
        typer.secho("\n⚠️  Splits already exist — skipping.", fg=typer.colors.YELLOW)
        raise typer.Exit()

    if has_train and has_test:
        typer.secho("\n📂 Found train/test files — moving to splits/", fg=typer.colors.CYAN)
        shutil.copy(
            os.path.join(proc_dir, "train.parquet"), out_train
        )
        shutil.copy(
            os.path.join(proc_dir, "test.parquet"), out_test
        )
        typer.secho("✅ Done (used provided split).", fg=typer.colors.GREEN)
        raise typer.Exit()

    # otherwise we need to sample
    # pick the main dataset file
    candidates = [pq for pq in parquets if pq != "sample_submission.parquet"]
    if len(candidates) == 0:
        typer.secho("❌ No dataset found to split!", fg=typer.colors.RED)
        raise typer.Exit(1)
    src_pq = (
        os.path.join(proc_dir, "train.parquet")
        if "train.parquet" in candidates
        else os.path.join(proc_dir, candidates[0])
    )
    typer.secho("\n✂️  Splitting off test set …", fg=typer.colors.CYAN)
    df = pd.read_parquet(src_pq)
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    train.to_parquet(out_train, index=False)
    test.to_parquet(out_test,  index=False)
    typer.secho(
        f"✅ Created splits → train: {train.shape}, test: {test.shape}",
        fg=typer.colors.GREEN
    )

if __name__ == "__main__":
    app()