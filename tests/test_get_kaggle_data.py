import pytest
from unittest.mock import patch, MagicMock, call
import typer
import os

# Import the CLI app and helpers
def import_main():
    # Import inside function to avoid side effects
    from src import get_kaggle_data
    return get_kaggle_data

@patch('src.get_kaggle_data.os.makedirs')
@patch('src.get_kaggle_data.os.listdir')
@patch('src.get_kaggle_data.typer.secho')
@patch('src.get_kaggle_data.typer.echo')
def test_main_skips_download(mock_echo, mock_secho, mock_listdir, mock_makedirs):
    get_kaggle_data = import_main()
    # Simulate CSVs already present
    mock_listdir.return_value = ['train.csv', 'test.csv']
    with patch('src.get_kaggle_data.pd.read_csv'), \
         patch('src.get_kaggle_data.pd.read_parquet'), \
         patch('src.get_kaggle_data.pd.DataFrame.to_parquet'), \
         patch('src.get_kaggle_data.dir_size', return_value='1.0MiB'), \
         patch('src.get_kaggle_data.human_size', return_value='1.0MiB'), \
         patch('src.get_kaggle_data.shutil.copy') as mock_copy:
        # Should skip download and exit after reporting splits
        with pytest.raises(typer.Exit):
            get_kaggle_data.main('m5-forecasting-accuracy')
        mock_secho.assert_any_call('⚠️  CSVs already in place — skipping download.', fg=typer.colors.YELLOW)

@patch('src.get_kaggle_data.os.makedirs')
@patch('src.get_kaggle_data.os.listdir')
@patch('src.get_kaggle_data.subprocess.run')
@patch('src.get_kaggle_data.zipfile.ZipFile')
@patch('src.get_kaggle_data.typer.secho')
@patch('src.get_kaggle_data.typer.echo')
def test_main_downloads_and_unzips(mock_echo, mock_secho, mock_zip, mock_run, mock_listdir, mock_makedirs):
    get_kaggle_data = import_main()
    # Simulate no CSVs, but a zip file present after download
    mock_listdir.side_effect = [[], ['data.zip'], ['train.csv']]
    mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='')
    with patch('src.get_kaggle_data.pd.read_csv'), \
         patch('src.get_kaggle_data.pd.read_parquet'), \
         patch('src.get_kaggle_data.pd.DataFrame.to_parquet'), \
         patch('src.get_kaggle_data.dir_size', return_value='1.0MiB'), \
         patch('src.get_kaggle_data.human_size', return_value='1.0MiB'), \
         patch('src.get_kaggle_data.os.remove'), \
         patch('src.get_kaggle_data.typer.Option', side_effect=lambda *a, **kw: 0.2):
        # Should attempt download and unzip
        mock_zip.return_value.__enter__.return_value.extractall = MagicMock()
        with pytest.raises(typer.Exit):
            get_kaggle_data.main('m5-forecasting-accuracy')
        mock_secho.assert_any_call('⏬ Downloading m5-forecasting-accuracy', fg=typer.colors.CYAN)
        assert mock_run.called
        assert mock_zip.called
