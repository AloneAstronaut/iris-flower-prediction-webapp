import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Drop remote_users table if it exists
    cursor.execute('DROP TABLE IF EXISTS remote_users')

    # Create remote_users table
    cursor.execute('''
        CREATE TABLE remote_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            last_login DATETIME NOT NULL
        )
    ''')

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
    except sqlite3.IntegrityError:
        return False  # User already exists
    finally:
        conn.close()
    return True

def validate_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

def add_remote_user(username, ip_address):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Insert new user with current timestamp
    cursor.execute('''
        INSERT INTO remote_users (username, ip_address, last_login) 
        VALUES (?, ?, ?)
    ''', (username, ip_address, datetime.now()))

    conn.commit()
    conn.close()

def fetch_remote_users():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username, ip_address, last_login FROM remote_users')
    users = cursor.fetchall()
    conn.close()
    return [{'username': user[0], 'ip': user[1], 'last_login': user[2]} for user in users]
