import mysql.connector as con
import io
import gzip
import streamlit as st
from passlib.hash import pbkdf2_sha256
database=con.connect(        host="DB_HOST",
        user="DB_USER",
        password="DB_PASS",
        database="DB_NAME")
print(database.connection_id)

cursor=database.cursor()

# cursor.execute("CREATE DATABASE LDAPROJECT")

# cursor.execute("CREATE TABLE user_login (UserId INT AUTO_INCREMENT PRIMARY KEY, username VARCHAR(255), password VARCHAR(255))")
# cursor.execute("CREATE TABLE pdf_storage (DataId INT AUTO_INCREMENT PRIMARY KEY, UserId INT NOT NULL, filename VARCHAR(255), pdf_data LONGBLOB, pyldavis_html LONGBLOB, FOREIGN KEY (UserId) REFERENCES user_login (UserId))")
# cursor.execute("ALTER TABLE PDF_STORAGE ADD COLUMN pyldavis_html LONGBLOB")
# cursor.execute("ALTER TABLE PDF_STORAGE DROP COLUMN HTML")
database.commit()

# database.close()



def connect_to_database():
    return con.connect(
        host="DB_HOST",
        user="DB_USER",
        password="DB_PASS",
        database="DB_NAME"
    )

def register_user(username, password,mydb):
    hashed_password = pbkdf2_sha256.hash(password)
    cursor = mydb.cursor()
    insert_query = "INSERT INTO user_login (username, password) VALUES (%s, %s)"
    cursor.execute(insert_query, (username, hashed_password))
    mydb.commit()
    cursor.close()


def verify_login(mydb,username,password):
    cursor=mydb.cursor()
    cursor.execute("SELECT PASSWORD FROM USER_LOGIN WHERE username=%s",(username,))
    user_data=cursor.fetchone()
    if user_data:
        hashed_password = user_data[0]
        return pbkdf2_sha256.verify(password, hashed_password)
    return False




def insert_pdf_into_database(filename, pdf_data, html_data,mydb,userid):    
    cursor=mydb.cursor()
    if pdf_exists(filename,mydb):
        update_query = "UPDATE pdf_storage SET pyldavis_html = %s WHERE filename = %s"
        val = (html_data, filename)
        cursor.execute(update_query, val)
    else:
        sql = "INSERT INTO pdf_storage (filename, pdf_data, pyldavis_html,UserId) VALUES (%s, %s,%s,%s)"
        val = (filename, pdf_data,html_data,userid)
        cursor.execute(sql, val)
    mydb.commit()
    # mydb.close()


def retrieve_data(mydb,userid):
    cursor=mydb.cursor()
    cursor.execute("SELECT * FROM pdf_storage WHERE UserId = %s", (userid,))
    data=cursor.fetchall()
    # if st.button("Delete"):
    #     cursor.execute("DELETE FROM pdf_storage WHERE UserId = %s", (userid,))
    #     st.rerun()
    #     mydb.commit()
    return data


def pdf_exists(pdf_name,mydb):
    cursor = mydb.cursor()
    cursor.execute("SELECT COUNT(*) FROM pdf_storage WHERE filename = %s", (pdf_name,))
    count = cursor.fetchone()[0]
    cursor.close()
    return count > 0




def get_user_id(username,mydb):
    cursor = mydb.cursor()
    cursor.execute("SELECT UserId FROM user_login WHERE username = %s", (username,))
    user_id = cursor.fetchone()[0]
    cursor.close()
    # print(user_id)
    return user_id
