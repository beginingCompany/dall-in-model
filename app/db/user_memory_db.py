import sqlite3

DB_PATH = "user_memory.db"

def get_db_conn():
    return sqlite3.connect(DB_PATH)

def create_table():
    with get_db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory (
                id INTEGER PRIMARY KEY,
                user_input TEXT,
                new_input TEXT
            );
            """
        )

def load_user_memory(user_id: int):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_input, new_input FROM user_memory WHERE id=?", (user_id,))
        row = cur.fetchone()
        if row:
            return {"user_input": row[0], "new_input": row[1]}
        else:
            return {"user_input": "", "new_input": ""}

def save_user_memory(user_id: int, user_input: str, new_input: str):
    with get_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO user_memory (id, user_input, new_input)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET user_input=excluded.user_input, new_input=excluded.new_input
            """,
            (user_id, user_input, new_input)
        )
