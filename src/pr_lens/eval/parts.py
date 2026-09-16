"""How many parts each model's corpus and query sets were cut into.

Anything that reads a stored part has to agree with whatever wrote it: the embed job, the
recall harness and review retrieval all index the same files. So the numbers live on their
own, in a module that imports nothing, because jobs/plan.py builds the workflow matrices
from them and its runner installs no dependency groups at all.

Sized from CPU throughput measured 12 Sep 2026: bge-small 11 to 15 texts a second,
gte-modernbert 0.45 to 4 depending on length, every matrix job well inside an hour.
"""

EMBED_PARTS = {"bge-small": 1, "gte-modernbert": 8}
QUERY_PARTS = {"bge-small": 1, "gte-modernbert": 4}
