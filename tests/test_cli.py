import subprocess

def test_cli_runs():
    result = subprocess.run(['python', 'src/cli.py', '--help'], capture_output=True)
    assert result.returncode == 0