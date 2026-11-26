"""Out-of-core data: write training data to disk in shards and stream it back in bounded memory.

The machine this was built on has 5.9 GB of RAM and hundreds of GB free on disk, so data never has to fit
in memory: records and arrays go to shards under ``AEROSYNAPSE_STORE`` (default ``<project>/data_store``)
with a manifest and checksums, and training reads them back shard by shard through a shuffle buffer.
"""
