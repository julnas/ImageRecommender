import sqlite3
from typing import Optional, List

class Database:
    def __init__(self, db_path: str):
        """
        Constructor that connects to the SQLite database.

        Parameters:
        db_path: path to the database file
        """
        self.connection = sqlite3.connect(db_path)  #connect to the SQLite DB file  -> SQLite safes how i can find a pic and the things i know about the  pic
        self.cursor = self.connection.cursor()       #create a cursor to execute SQL commands

    def get_image_path(self, image_id: int):
        """
        Retrieves the file path of an image based on its image ID.

        Parameters:
        image_id: ID of the image

        Returns:
        The file path as a string
        """
        self.cursor.execute("SELECT file_path FROM images WHERE image_id = ?", (image_id,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_all_image_ids(self):
        """
        Retrieves all image IDs stored in the database.

        Returns:
        A list of integers representing image IDs
        """
        self.cursor.execute("SELECT image_id FROM images")
        return [row[0] for row in self.cursor.fetchall()]

    def close(self):
        """
        Closes the database connection.
        """
        self.connection.close()
