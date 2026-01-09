DROP TABLE IF EXISTS coins;

CREATE TABLE coins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    time INTEGER NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL
);
