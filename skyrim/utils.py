from dotenv import load_dotenv
from pathlib import Path
import os
from loguru import logger


def ensure_ecmwf_loaded():
    """
    To be able to use ecmwf's API for IFS, etc.
    we need to have the config file in /root/.ecmwfapirc
    """

    config_path = Path("~/.ecmwfapirc").expanduser()
    if config_path.exists():
        return True

    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    ecmwf_key = os.environ.get("ECMWF_API_KEY")
    ecmwf_url = os.environ.get("ECMWF_API_URL")
    ecmwf_email = os.environ.get("ECMWF_API_EMAIL")
    logger.info(f"Gathering ECMWF API key from environment...")

    if not all([ecmwf_key, ecmwf_url, ecmwf_email]):
        raise Exception("ECMWF API config not found in the environment.")

    config_content = f'{{\n  "url": "{ecmwf_url}",\n  "key": "{ecmwf_key}",\n  "email": "{ecmwf_email}"\n}}'
    config_path.write_text(config_content)
    logger.info("ECMWF API config file created at /root/.ecmwfapirc.")


def ensure_cds_loaded():
    """Currently, earth2mip requires CDS env to be loaded in /root/.cdsapi"""
    if Path("~/.cdsapirc").expanduser().exists():
        return True
    else:
        load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
        cds_key = os.environ.get("CDSAPI_KEY")
        cds_url = os.environ.get("CDSAPI_URL")
        logger.info(f"Gathering CDS API key from environment...")
        if not cds_key:
            raise Exception("CDS API config not found in the environment.")
        Path("~/.cdsapirc").write_text(f"key: {cds_key}\nurl: {cds_url}")
        logger.success(f"Successfully wrote CDS API key to /root/.cdsapi")


if __name__ == "__main__":
    print(Path(__file__).parent.parent / ".env")
