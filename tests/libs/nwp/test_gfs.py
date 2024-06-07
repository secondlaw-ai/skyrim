import pytest
import math
from skyrim.libs.nwp.gfs import GFS_Vocabulary


def test_vocab_attribute():
    """Tests that the VOCAB attribute is populated correctly."""
    assert GFS_Vocabulary.VOCAB is not None


def test_direct_access():
    """Tests direct access to GFS_Vocabulary without initialization."""
    GFS_VOCAB = GFS_Vocabulary()
    assert GFS_VOCAB["u10m"] == "UGRD::10 m above ground"


def test_get_valid_channel():
    """Tests the get() method for valid channels."""
    gfs_id, gfs_level, modifier_func = GFS_Vocabulary.get("u100")
    assert gfs_id == "UGRD"
    assert gfs_level == "100 mb"
    assert modifier_func(10) == 10  # Test identity function


def test_get_hgt_channel():
    """Tests the get() method for HGT channels (special case)."""
    gfs_id, gfs_level, modifier_func = GFS_Vocabulary.get("z250")
    assert gfs_id == "HGT"
    assert gfs_level == "250 mb"
    assert math.isclose(modifier_func(10), 98.1)  # Test gravity conversion


def test_get_invalid_channel():
    """Tests that an exception is raised for invalid channels."""
    with pytest.raises(KeyError):
        GFS_Vocabulary.get("invalid_channel")


def test_contains_method():
    """Tests the __contains__ method."""
    GFS_VOCAB = GFS_Vocabulary()
    assert "t2m" in GFS_VOCAB
    assert "invalid_key" not in GFS_VOCAB
