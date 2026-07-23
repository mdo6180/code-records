import sqlite3
from pathlib import Path


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def copy_all_tables(high_side_db: Path, low_side_db: Path) -> None:

    with sqlite3.connect(high_side_db) as connection:
        connection.execute(
            "ATTACH DATABASE ? AS low",
            (str(low_side_db),),
        )

        try:
            # extract the table names and their creation SQL statements from the low side database
            tables = connection.execute(
                """
                SELECT name, sql
                FROM low.sqlite_schema
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                  AND sql IS NOT NULL
                ORDER BY name
                """
            ).fetchall()

            connection.execute("BEGIN IMMEDIATE")

            for table_name, create_sql in tables:
                # format the table name to be safely used in SQL statements
                quoted_table = quote_identifier(table_name)

                # check if the table already exists in the main database
                exists = connection.execute(
                    """
                    SELECT 1
                    FROM main.sqlite_schema
                    WHERE type = 'table'
                      AND name = ?
                    """,
                    (table_name,),
                ).fetchone()

                # if the table does not exist, create it in the main database
                if exists is None:
                    connection.execute(create_sql)

                # copy the data from the low side table to the main database table
                connection.execute(
                    f"""
                    INSERT OR IGNORE INTO main.{quoted_table}
                    SELECT *
                    FROM low.{quoted_table}
                    """
                )

            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.execute("DETACH DATABASE low")


if __name__ == "__main__":
    high_side_db = Path(__file__).parent / "high_side.db"
    low_side_db = Path(__file__).parent / "low_side.db"

    copy_all_tables(high_side_db, low_side_db)