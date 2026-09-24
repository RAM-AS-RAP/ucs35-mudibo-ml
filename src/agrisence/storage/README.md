# Storage

This folder is reserved for persistence helpers and future data writes. The current implementation intentionally avoids real Supabase connection logic because the project schema is not yet created, and the work is staged to happen through migrations in a separate step.

## Files

- `supabase_client.py` — placeholder client for feature, market-price, and forecast upserts after schema creation.
- `__init__.py` — marks the package for import.
