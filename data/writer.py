import os
import uuid
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq


class ParquetShardWriter:
    def __init__(
        self,
        out_dir: str,
        table_name: str,
        schema: pa.schema,
        rows_per_shard: int = 50_000,
        compression: str = "zstd",
        use_dictionary: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.table_name = table_name
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.compression = compression
        self.use_dictionary = use_dictionary

        os.makedirs(self.out_dir, exist_ok=True)

        self._shard_id = 0
        self._row_count_in_shard = 0
        self._writer: Optional[pq.ParquetWriter] = None
        self._open_new_shard()

    def _shard_path(self, shard_id: int) -> str:
        return os.path.join(self.out_dir, f"{self.table_name}-{shard_id:06d}.parquet")

    def _open_new_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._row_count_in_shard = 0
        self._shard_id += 1
        # write to temp first for atomicity
        final_path = self._shard_path(self._shard_id)
        tmp_path = final_path + f".{uuid.uuid4().hex}.tmp"
        self._current_tmp_path = tmp_path
        self._final_path = final_path
        self._writer = pq.ParquetWriter(
            where=tmp_path,
            schema=self.schema,
            compression=self.compression,
            use_dictionary=self.use_dictionary,
        )

    def append_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        # validate schema compatibility
        if not table.schema.equals(self.schema, check_metadata=False):
            table = table.cast(self.schema)
        self._writer.write_table(table)
        self._row_count_in_shard += table.num_rows
        if self._row_count_in_shard >= self.rows_per_shard:
            self.rotate_shard()

    def rotate_shard(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        os.replace(self._current_tmp_path, self._final_path)
        self._writer = None
        self._open_new_shard()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            os.replace(self._current_tmp_path, self._final_path)
            self._writer = None


def table_from_pydict(schema: pa.schema, data: dict) -> pa.Table:
    # Use positional args for PyArrow 12 / Python 3.7 compatibility
    return pa.Table.from_pydict(data, schema)


class ParquetDatasetWriter:
    """
    Simple dataset writer that creates a finalized Parquet file per append.
    This avoids temp file renames and works robustly on Windows.
    Files are written under out_dir/table_name/part-<uuid>.parquet
    """

    def __init__(
        self,
        out_dir: str,
        table_name: str,
        schema: pa.schema,
        compression: str = "snappy",
    ) -> None:
        self.base_dir = os.path.join(out_dir, table_name)
        os.makedirs(self.base_dir, exist_ok=True)
        self.schema = schema
        self.compression = compression

    def append_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if not table.schema.equals(self.schema, check_metadata=False):
            table = table.cast(self.schema)
        path = os.path.join(self.base_dir, f"part-{uuid.uuid4().hex}.parquet")
        pq.write_table(table, path, compression=self.compression)

    def close(self) -> None:
        # No-op; files are already finalized on each append
        return


class DuckDBWriter:
    """
    Append-only writer backed by DuckDB. Each table is created once and appended via Arrow.
    On finalize, tables can be exported to Parquet files for downstream consumption.
    """
    def __init__(
        self,
        out_dir: str,
        table_name: str,
        schema: pa.schema,
        db_path: str,
    ) -> None:
        import duckdb  # local import to keep import-time light

        self.table_name = table_name
        self.schema = schema
        os.makedirs(out_dir, exist_ok=True)
        self._con = duckdb.connect(db_path)
        self._ensure_table()

    def _ensure_table(self) -> None:
        # Map Arrow types to DuckDB SQL
        def arrow_to_sql(f: pa.Field) -> str:
            t = f.type
            if pa.types.is_string(t):
                return "VARCHAR"
            if pa.types.is_int8(t):
                return "TINYINT"
            if pa.types.is_int16(t):
                return "SMALLINT"
            if pa.types.is_int32(t):
                return "INTEGER"
            if pa.types.is_int64(t):
                return "BIGINT"
            if pa.types.is_float16(t) or pa.types.is_float32(t):
                return "FLOAT"
            if pa.types.is_float64(t):
                return "DOUBLE"
            if pa.types.is_binary(t) or pa.types.is_large_binary(t):
                return "BLOB"
            if pa.types.is_list(t):
                # DuckDB requires LIST with concrete element type, e.g., DOUBLE[]
                elem = t.value_type
                if pa.types.is_float64(elem) or pa.types.is_float32(elem) or pa.types.is_float16(elem):
                    return "DOUBLE[]"
                if pa.types.is_int64(elem):
                    return "BIGINT[]"
                if pa.types.is_int32(elem) or pa.types.is_int16(elem) or pa.types.is_int8(elem):
                    return "INTEGER[]"
                if pa.types.is_string(elem):
                    return "VARCHAR[]"
                # Fallback to JSON-like representation
                return "VARCHAR"
            return "VARCHAR"

        cols = ", ".join([f"{f.name} {arrow_to_sql(f)}" for f in self.schema])
        self._con.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} ({cols});")

    def append_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if not table.schema.equals(self.schema, check_metadata=False):
            table = table.cast(self.schema)
        self._con.register("tmp_arrow", table)
        self._con.execute(f"INSERT INTO {self.table_name} SELECT * FROM tmp_arrow;")
        self._con.unregister("tmp_arrow")

    def row_count(self) -> int:
        return int(self._con.execute(f"SELECT COUNT(*) FROM {self.table_name};").fetchone()[0])

    def export_to_parquet_dir(self, base_out_dir: str, compression: str = "gzip") -> None:
        out_dir = os.path.join(base_out_dir, self.table_name)
        os.makedirs(out_dir, exist_ok=True)
        comp = (compression or "gzip").upper()
        self._con.execute(
            f"COPY {self.table_name} TO '{out_dir}/part-*.parquet' (FORMAT PARQUET, COMPRESSION '{comp}', PER_THREAD_OUTPUT TRUE);"
        )

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

