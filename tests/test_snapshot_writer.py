import polars as pl

from backend.collector.writer import SnapshotWriter


async def test_snapshot_writer_flushes_append_only_chunks(tmp_path):
    writer = SnapshotWriter(tmp_path, "UBUSDT", batch_rows=1)

    await writer.append({"ts_ms": 1, "symbol": "UBUSDT", "x": 1.0})
    await writer.append({"ts_ms": 2, "symbol": "UBUSDT", "x": 2.0})
    await writer.close()

    files = sorted((tmp_path / "snapshots" / "UBUSDT").glob("*/*.parquet"))

    assert len(files) == 2
    assert all(file.name != "00.parquet" for file in files)
    df = pl.concat([pl.read_parquet(file) for file in files])
    assert sorted(df["ts_ms"].to_list()) == [1, 2]
