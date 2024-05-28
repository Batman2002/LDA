# import mysql.connector as con
from google.cloud.sql.connector import Connector,IPTypes
import sqlalchemy
import pymysql

# import firebase_admin
# from firebase_admin import credentials,auth
import io
import gzip
import streamlit as st
from passlib.hash import pbkdf2_sha256


# cred=credentials.Certificate("ldaproject-5648f-79beb07cda08.json")

# firebase_admin.initialize_app(cred)
# database=Connector.connect(host='localhost', 
#                      user='root',
#                      password='Bananasmoothie@123',
#                      database="LDAPROJECT")
# print(database.connection_id)

# cursor=database.cursor()

# cursor.execute("CREATE DATABASE LDAPROJECT")

# cursor.execute("CREATE TABLE user_login (UserId INT AUTO_INCREMENT PRIMARY KEY, username VARCHAR(255), password VARCHAR(255))")
# cursor.execute("CREATE TABLE pdf_storage (DataId INT AUTO_INCREMENT PRIMARY KEY, UserId INT NOT NULL, filename VARCHAR(255), pdf_data LONGBLOB, pyldavis_html LONGBLOB, FOREIGN KEY (UserId) REFERENCES user_login (UserId))")
# cursor.execute("ALTER TABLE PDF_STORAGE ADD COLUMN pyldavis_html LONGBLOB")
# cursor.execute("ALTER TABLE PDF_STORAGE DROP COLUMN HTML")
# database.commit()
# 
# database.close()

connector=Connector(ip_type=IPTypes.PUBLIC)

def connect_to_database():
    con: pymysql.connections.Connection=connector.connect(
        "totemic-tower-424510-q7:us-central1:trial-sql",
        "pymysql",
        user="main",
        password="Bananasmoothie@123",
        db="LDAPROJECT",
        # cursorclass=pymysql.cursors.DictCursor
    )
    return con

def get_connection():
    pool=sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=connect_to_database
    )
    return pool


def register_user(username, password):
    hashed_password = pbkdf2_sha256.hash(password)
    # cursor = mydb.cursor()
    engine=get_connection()
    # parameters={"username":username, "password":passw}
    with engine.connect() as cursor:
        insert_query = """INSERT INTO user_login (username, password) VALUES (%s, %s);"""
        cursor.execute(insert_query, (username, hashed_password))
        # cursor.commit()
    # cursor.close()


def verify_login(username,password):
    engine=get_connection()
    with engine.connect() as cursor:
        user_data=cursor.execute("SELECT PASSWORD FROM user_login WHERE username=%s",(username,)).fetchone()
        if user_data:
            hashed_password = user_data[0]
            return pbkdf2_sha256.verify(password, hashed_password)
    return False




def insert_pdf_into_database(filename, pdf_data, html_data,mydb,userid):    
    engine=get_connection()
    with engine.connect() as cursor:
        if pdf_exists(filename,mydb):
            update_query = "UPDATE pdf_storage SET pyldavis_html = %s WHERE filename = %s"
            val = (html_data, filename)
            cursor.execute(update_query, val)
        else:
            sql = "INSERT INTO pdf_storage (filename, pdf_data, pyldavis_html,UserId) VALUES (%s, %s,%s,%s)"
            val = (filename, pdf_data,html_data,userid)
            cursor.execute(sql, val)
        # mydb.commit()
    # mydb.close()


def retrieve_data(mydb,userid):
    engine=get_connection()
    with engine.connect() as cursor:
        data=cursor.execute("SELECT * FROM pdf_storage WHERE UserId = %s", (userid,)).fetchall()
        # data=cursor.fetchall()
    # if st.button("Delete"):
    #     cursor.execute("DELETE FROM pdf_storage WHERE UserId = %s", (userid,))
    #     st.rerun()
    #     mydb.commit()
    return data


def pdf_exists(pdf_name,mydb):
    engine=get_connection()
    with engine.connect() as cursor:
        data=cursor.execute("SELECT COUNT(*) FROM pdf_storage WHERE filename = %s", (pdf_name,)).fetchone()
        count = data[0]
        # cursor.close()
        return count > 0




def get_user_id(username,mydb):
    engine=get_connection()
    with engine.connect() as cursor:
        output=cursor.execute("SELECT UserId FROM user_login WHERE username = %s", (username,)).fetchone()
        user_id = output[0]
    # cursor.close()
    # print(user_id)
    return user_id