from config import get_settings
from neo4j import GraphDatabase

s = get_settings()

driver = GraphDatabase.driver(
    s.neo4j_uri,
    auth=(s.neo4j_username, s.neo4j_password)
)

try:
    with driver.session(database=s.neo4j_database) as session:
        result = session.run("RETURN 'Neo4j connected successfully' AS msg")
        print(result.single()["msg"])
finally:
    driver.close()