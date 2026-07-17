"""
src/utils/secrets.py
--------------------
Cross-platform secret resolution.

Priority order:
  1. Streamlit st.secrets  (Streamlit Community Cloud)
  2. os.environ            (HF Spaces, local .env via python-dotenv)
  3. default value         (empty string, so callers can detect missing keys)

Usage:
    from src.utils.secrets import get_secret
    api_key = get_secret("GROQ_API_KEY")
"""

import os

from dotenv import load_dotenv

# Load .env for local development (no-op if file doesn't exist)
load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """
    Retrieve a secret by key, checking Streamlit secrets first,
    then environment variables.

    The try/except around st.secrets is intentional: on HF Spaces and
    in plain Python scripts, Streamlit is not running, so st.secrets
    raises FileNotFoundError or RuntimeError — both are safely caught here.
    """
    try:
        import streamlit as st  # lazy import — don't break scripts without streamlit

        # st.secrets[key] raises KeyError if key is absent,
        # FileNotFoundError if no secrets file exists (local dev without streamlit run)
        return st.secrets[key]
    except Exception:
        pass

    return os.environ.get(key, default)
