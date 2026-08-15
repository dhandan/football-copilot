import duckdb


connection = duckdb.connect("database/football.duckdb")


result = connection.execute("""
    SELECT *
    FROM matches
    LIMIT 10
""").fetchdf()


print(result)


connection.close()