"""
DuckDB connection and query manager.
Provides high-performance analytical queries directly over Parquet files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import duckdb
import pandas as pd


class DuckDBManager:
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize DuckDB connection.
        If db_path is ':memory:' or None, runs an in-memory instance.
        """
        if db_path and str(db_path) != ":memory:":
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = duckdb.connect(str(self.db_path))
        else:
            self.db_path = None
            self.conn = duckdb.connect(":memory:")

    def execute(self, sql: str, params: Optional[Union[List[Any], Dict[str, Any]]] = None) -> Any:
        """Execute a query without returning result dataframe."""
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)

    def query_df(self, sql: str, params: Optional[Union[List[Any], Dict[str, Any]]] = None) -> pd.DataFrame:
        """Execute SQL query and return result as pandas DataFrame."""
        if params:
            return self.conn.execute(sql, params).df()
        return self.conn.execute(sql).df()

    def register_parquet(self, view_name: str, parquet_path: Union[str, Path]) -> None:
        """Register a parquet file or directory as a virtual SQL table/view."""
        path_str = str(parquet_path).replace("\\", "/")
        sql = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{path_str}')"
        self.conn.execute(sql)

    def register_df(self, view_name: str, df: pd.DataFrame) -> None:
        """Register a pandas dataframe as a temporary view."""
        self.conn.register(view_name, df)

    def export_to_parquet(self, query: str, output_path: Union[str, Path]) -> None:
        """Export SQL query results directly to Parquet file."""
        out_str = str(output_path).replace("\\", "/")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sql = f"COPY ({query}) TO '{out_str}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        self.conn.execute(sql)

    def get_table_names(self) -> List[str]:
        """List all tables and views in DuckDB catalog."""
        df = self.query_df("SHOW TABLES")
        if not df.empty:
            return df["name"].tolist()
        return []

    def close(self) -> None:
        """Close connection."""
        if self.conn:
            self.conn.close()
