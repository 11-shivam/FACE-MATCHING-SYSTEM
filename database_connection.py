import mysql.connector

def get_connection() :
    connection = mysql.connector.connect(
        host = "127.0.0.1",
        user = "root",
        password = "JK666J3L3K",
        database = "faceEmbedding"
    )

    return connection