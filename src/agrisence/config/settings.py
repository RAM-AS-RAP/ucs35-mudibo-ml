"""Environment settings for local model execution and pipeline configuration.

This module loads values from a local .env file so credentials stay out of source control.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# TODO: populate .env once Supabase project is provisioned
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
OPENMETEO_API_KEY = os.getenv("OPENMETEO_API_KEY", "")

