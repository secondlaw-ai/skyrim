import pytest
from pathlib import Path
from skyrim.forecast import main
from click.testing import CliRunner

@pytest.mark.integ
def test_forecasts():
    runner = CliRunner()
    res = runner.invoke(main)
    assert res.exit_code == 0
    