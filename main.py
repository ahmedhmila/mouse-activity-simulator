import sys
import random
import time
import threading
import string
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QSystemTrayIcon, QMenu, QAction, QApplication, 
                            QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QSlider, QCheckBox, QGroupBox, QWidget,
                            QProgressBar, QSpinBox, QComboBox, QStyleFactory,
                            QRadioButton)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QRect
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette
import pyautogui
import keyboard

class MouseActivitySimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.running = False
        self.timer = None
        self.next_action_name = "None"
        self.elapsed_seconds = 0 
        
        # Python keywords and functions for typing
        self.python_code_snippets = [
            # Basic imports
            "import os", "import sys", "import time", "import datetime", "import random", 
            "import json", "import csv", "import pickle", "import logging", "import re",
            "import glob", "import shutil", "import argparse", "import configparser",
            "import multiprocessing", "import threading", "import concurrent.futures",
            
            # Data analysis imports
            "import numpy as np", "import pandas as pd", "import matplotlib.pyplot as plt", 
            "import seaborn as sns", "import plotly.express as px", "import scipy.stats as stats",
            
            # Database imports
            "import sqlite3", "import psycopg2", "import pymysql", "import pyodbc", 
            "import sqlalchemy", "from sqlalchemy import create_engine", "import pymongo",
            
            # Big data imports
            "import pyspark", "from pyspark.sql import SparkSession", "import dask", 
            "import dask.dataframe as dd", "import vaex", "import ray",
            
            # Cloud imports
            "import boto3", "import google.cloud", "import azure.storage.blob",
            "from google.cloud import storage", "from google.cloud import bigquery",
            
            # Web/API imports
            "import requests", "import urllib", "import aiohttp", "import fastapi", 
            "import flask", "from flask import Flask, request, jsonify",
            
            # Data formats
            "import pyarrow", "import pyarrow.parquet as pq", "import pyarrow.csv as csv",
            "import avro", "import fastavro", "import orjson", "import msgpack",
            
            # Machine learning
            "import sklearn", "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler", "import tensorflow as tf",
            
            # Basic statements
            "def function():", "class MyClass():", "if condition == True:", "elif condition:", 
            "else:", "for i in range(10):", "while True:", "break", "continue", "pass",
            
            # Exception handling
            "try:", "except Exception as e:", "except (TypeError, ValueError) as e:", 
            "finally:", "raise Exception('Error message')", "assert condition, 'message'",
            
            # Output and return
            "print('Hello, World!')", "return result", "yield item", 
            "self.variable = value", "logging.info('Process started')",
            "logging.error(f'Error: {e}')", "print(f'Value: {value}')",
            
            # File operations
            "with open('file.txt', 'r') as f:", "with open('file.txt', 'w') as f:", 
            "f.read()", "f.readlines()", "f.write('text')", "os.path.join(dir, file)",
            "os.path.exists(path)", "os.makedirs('dir', exist_ok=True)",
            "glob.glob('*.csv')", "for file in os.listdir(directory):",
            
            # Python data structures
            "[x for x in range(10)]", "{'key': 'value'}", "set([1, 2, 3])",
            "{x: x*2 for x in range(5)}", "(x for x in range(10))", 
            "lambda x: x*2", "sorted(items, key=lambda x: x['value'])",
            "list(map(func, items))", "list(filter(lambda x: x > 0, items))",
            "random.choice(items)", "dict(zip(keys, values))", "from collections import defaultdict",
            "from collections import Counter", "from collections import deque",
            
            # Numpy operations
            "np.array([1, 2, 3])", "np.zeros((3, 3))", "np.ones((3, 3))", 
            "np.random.rand(10)", "np.random.normal(0, 1, 100)", "np.mean(array)",
            "np.median(array)", "np.std(array)", "np.sum(array)", "np.argmax(array)",
            "np.concatenate([arr1, arr2])", "np.reshape(array, (3, 3))",
            "np.where(condition, x, y)", "np.histogram(data, bins=10)",
            
            # Pandas operations
            "df = pd.DataFrame()", "df = pd.read_csv('file.csv')", "df.head()", 
            "df.describe()", "df.info()", "df.columns", "df.shape", "df.dtypes",
            "df['column']", "df[df['column'] > 10]", "df.iloc[0:10, 1:3]", 
            "df.loc[:, ['col1', 'col2']]", "df.fillna(0)", "df.dropna()",
            "df.drop_duplicates()", "df.sort_values(by='column')", 
            "df.rename(columns={'old': 'new'})", "df.groupby('group_col').agg({'val': 'sum'})",
            "df.merge(df2, on='key', how='left')", "df.join(df2)", "df.pivot_table(index='A', columns='B', values='C')",
            "df.melt(id_vars=['id'], value_vars=['A', 'B'])", "df.apply(func, axis=1)",
            "df['new_col'] = df['old_col'].apply(lambda x: x*2)", "df.query('col > 5 & col < 10')",
            "df.astype({'col': 'int'})", "df.to_csv('file.csv', index=False)",
            "df.to_parquet('file.parquet')", "pd.concat([df1, df2])",
            "pd.cut(df['col'], bins=5)", "pd.qcut(df['col'], q=4)",
            "pd.get_dummies(df['category'])", "pd.date_range(start='2023-01-01', periods=10)",
            
            # Spark operations
            "spark = SparkSession.builder.appName('app').getOrCreate()",
            "df = spark.read.csv('file.csv', header=True, inferSchema=True)",
            "df = spark.read.parquet('file.parquet')",
            "df.show()", "df.printSchema()", "df.select('col1', 'col2')", 
            "df.filter(df.col > 0)", "df.groupBy('col').agg({'val': 'sum'})",
            "df.withColumn('new_col', df.old_col * 2)", "df.join(df2, on='key')",
            "df.createOrReplaceTempView('table')",
            "spark.sql('SELECT * FROM table WHERE col > 5')",
            "df.write.mode('overwrite').parquet('output.parquet')",
            
            # Database operations
            "conn = sqlite3.connect('database.db')", "cursor = conn.cursor()",
            "conn = psycopg2.connect(host='localhost', database='db', user='user', password='pass')",
            "engine = create_engine('postgresql://user:pass@localhost/db')",
            "cursor.execute('SELECT * FROM table')", "results = cursor.fetchall()",
            "pd.read_sql('SELECT * FROM table', conn)",
            "pd.read_sql_query('SELECT * FROM table WHERE col = %s', conn, params=('value',))",
            "df.to_sql('table', conn, if_exists='replace', index=False)",
            "conn.commit()", "conn.close()",
            
            # API/Web operations
            "response = requests.get(url)", "response = requests.post(url, json=data)",
            "response.status_code", "response.json()", "response.text",
            "headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer token'}",
            "params = {'key1': 'value1', 'key2': 'value2'}",
            "response = requests.get(url, headers=headers, params=params)",
            
            # AWS operations
            "s3 = boto3.client('s3')", "s3.upload_file('file.txt', 'bucket-name', 'key')",
            "s3.download_file('bucket-name', 'key', 'file.txt')",
            "response = s3.list_objects_v2(Bucket='bucket-name', Prefix='prefix')",
            "athena = boto3.client('athena')",
            "glue = boto3.client('glue')",
            
            # JSON operations
            "with open('file.json', 'r') as f: data = json.load(f)",
            "with open('file.json', 'w') as f: json.dump(data, f, indent=4)",
            "json_str = json.dumps(data)", "data = json.loads(json_str)",
            
            # Time operations
            "start_time = time.time()", "elapsed = time.time() - start_time",
            "time.sleep(1)", "from datetime import datetime, timedelta",
            "now = datetime.now()", "today = datetime.today().date()",
            "yesterday = today - timedelta(days=1)",
            "dt = datetime.strptime('2023-01-01', '%Y-%m-%d')",
            "dt.strftime('%Y-%m-%d %H:%M:%S')",
            
            # Environment variables
            "os.environ.get('API_KEY')", "os.environ['DATABASE_URL']",
            "from dotenv import load_dotenv", "load_dotenv()",
            
            # Logging
            "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')",
            "logger = logging.getLogger(__name__)",
            "logger.info('Process started')", "logger.error('Error occurred: %s', e)",
            "handler = logging.FileHandler('app.log')",
            
            # Multiprocessing/Threading
            "with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:",
            "futures = [executor.submit(process_func, item) for item in items]",
            "results = [future.result() for future in concurrent.futures.as_completed(futures)]",
            "with concurrent.futures.ProcessPoolExecutor() as executor:",
            "pool = multiprocessing.Pool(processes=4)",
            "results = pool.map(process_func, items)",
            
            # Main execution
            "if __name__ == '__main__':", "def main():", "main()",
            
            # Pickle operations
            "with open('data.pkl', 'wb') as f: pickle.dump(obj, f)",
            "with open('data.pkl', 'rb') as f: obj = pickle.load(f)",
            
            # Configuration
            "config = configparser.ConfigParser()", "config.read('config.ini')",
            "value = config.get('section', 'key')",
            "parser = argparse.ArgumentParser(description='Process data')",
            "parser.add_argument('--input', required=True, help='Input file')",
            "args = parser.parse_args()"
        ]
        
        # Configuration settings with default values
        self.config = {
            'min_delay': 5,
            'max_delay': 7,
            'move_mouse': True,
            'scroll_screen': True,
            'alt_tab': True,
            'type_keys': True,
            'alt_tab_min': 2,
            'alt_tab_max': 5,
            # Frequency weights for each action (higher = more frequent)
            'move_weight': 10,
            'scroll_weight': 5,
            'alt_tab_weight': 3,
            'type_keys_weight': 3,
            'use_python_snippets': True
        }
        
        self.setup_ui()
        self.setup_tray()
        
    def setup_ui(self):
        # Set window properties
        self.setWindowTitle("Mouse Activity Simulator")
        self.setGeometry(100, 100, 500, 650)  
        app_icon = QIcon("src\\Icon.ico")
        self.setWindowIcon(app_icon)
        
        # Set dark theme
        self.apply_dark_theme()
        
        # Main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Timer display
        self.timer_layout = QHBoxLayout()
        self.timer_label = QLabel("Running Time: ")
        self.timer_value = QLabel("00:00:00")
        self.timer_value.setStyleSheet("font-weight: bold;")
        self.timer_layout.addWidget(self.timer_label)
        self.timer_layout.addWidget(self.timer_value)
        self.timer_layout.addStretch()
        main_layout.addLayout(self.timer_layout)

        

        # Title label
        title_label = QLabel("Mouse Activity Simulator by Ahmed Hmila")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 16, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Status indicator
        self.status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: ")
        self.status_value = QLabel("Stopped")
        self.status_value.setStyleSheet("color: red; font-weight: bold;")
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addWidget(self.status_value)
        self.status_layout.addStretch()
        main_layout.addLayout(self.status_layout)
        
        # Progress bar
        self.progress_group = QGroupBox("Next Action")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.next_action_label = QLabel("Waiting to start...")
        progress_layout.addWidget(self.next_action_label)
        progress_layout.addWidget(self.progress_bar)
        self.progress_group.setLayout(progress_layout)
        main_layout.addWidget(self.progress_group)
        
        # Options group box
        options_group = QGroupBox("Automation Options")
        options_layout = QVBoxLayout()
        
        # Checkboxes for actions
        self.move_checkbox = QCheckBox("Move Mouse")
        self.move_checkbox.setChecked(self.config['move_mouse'])
        options_layout.addWidget(self.move_checkbox)
        
        self.scroll_checkbox = QCheckBox("Scroll Screen")
        self.scroll_checkbox.setChecked(self.config['scroll_screen'])
        options_layout.addWidget(self.scroll_checkbox)
        
        self.alt_tab_checkbox = QCheckBox("Alt+Tab Between Windows")
        self.alt_tab_checkbox.setChecked(self.config['alt_tab'])
        options_layout.addWidget(self.alt_tab_checkbox)
        
        self.type_keys_checkbox = QCheckBox("Simulate Keyboard Activity")
        self.type_keys_checkbox.setChecked(self.config['type_keys'])
        self.type_keys_checkbox.setToolTip("Press random keys to simulate typing activity")
        options_layout.addWidget(self.type_keys_checkbox)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # NEW: Frequency settings
        frequency_group = QGroupBox("Action Frequency (Higher = More Frequent)")
        frequency_layout = QVBoxLayout()
        
        # Mouse move frequency
        move_freq_layout = QHBoxLayout()
        move_freq_layout.addWidget(QLabel("Move Mouse:"))
        self.move_freq_slider = QSlider(Qt.Horizontal)
        self.move_freq_slider.setRange(1, 20)
        self.move_freq_slider.setValue(self.config['move_weight'])
        self.move_freq_slider.setTickPosition(QSlider.TicksBelow)
        self.move_freq_slider.setTickInterval(1)
        self.move_freq_label = QLabel(str(self.config['move_weight']))
        self.move_freq_slider.valueChanged.connect(lambda v: self.move_freq_label.setText(str(v)))
        move_freq_layout.addWidget(self.move_freq_slider)
        move_freq_layout.addWidget(self.move_freq_label)
        frequency_layout.addLayout(move_freq_layout)
        
        # Scroll frequency
        scroll_freq_layout = QHBoxLayout()
        scroll_freq_layout.addWidget(QLabel("Scroll Screen:"))
        self.scroll_freq_slider = QSlider(Qt.Horizontal)
        self.scroll_freq_slider.setRange(1, 20)
        self.scroll_freq_slider.setValue(self.config['scroll_weight'])
        self.scroll_freq_slider.setTickPosition(QSlider.TicksBelow)
        self.scroll_freq_slider.setTickInterval(1)
        self.scroll_freq_label = QLabel(str(self.config['scroll_weight']))
        self.scroll_freq_slider.valueChanged.connect(lambda v: self.scroll_freq_label.setText(str(v)))
        scroll_freq_layout.addWidget(self.scroll_freq_slider)
        scroll_freq_layout.addWidget(self.scroll_freq_label)
        frequency_layout.addLayout(scroll_freq_layout)
        
        # Alt+Tab frequency
        alt_tab_freq_layout = QHBoxLayout()
        alt_tab_freq_layout.addWidget(QLabel("Alt+Tab:"))
        self.alt_tab_freq_slider = QSlider(Qt.Horizontal)
        self.alt_tab_freq_slider.setRange(1, 20)
        self.alt_tab_freq_slider.setValue(self.config['alt_tab_weight'])
        self.alt_tab_freq_slider.setTickPosition(QSlider.TicksBelow)
        self.alt_tab_freq_slider.setTickInterval(1)
        self.alt_tab_freq_label = QLabel(str(self.config['alt_tab_weight']))
        self.alt_tab_freq_slider.valueChanged.connect(lambda v: self.alt_tab_freq_label.setText(str(v)))
        alt_tab_freq_layout.addWidget(self.alt_tab_freq_slider)
        alt_tab_freq_layout.addWidget(self.alt_tab_freq_label)
        frequency_layout.addLayout(alt_tab_freq_layout)
        
        # Type keys frequency
        type_keys_freq_layout = QHBoxLayout()
        type_keys_freq_layout.addWidget(QLabel("Type Keys:"))
        self.type_keys_freq_slider = QSlider(Qt.Horizontal)
        self.type_keys_freq_slider.setRange(1, 20)
        self.type_keys_freq_slider.setValue(self.config['type_keys_weight'])
        self.type_keys_freq_slider.setTickPosition(QSlider.TicksBelow)
        self.type_keys_freq_slider.setTickInterval(1)
        self.type_keys_freq_label = QLabel(str(self.config['type_keys_weight']))
        self.type_keys_freq_slider.valueChanged.connect(lambda v: self.type_keys_freq_label.setText(str(v)))
        type_keys_freq_layout.addWidget(self.type_keys_freq_slider)
        type_keys_freq_layout.addWidget(self.type_keys_freq_label)
        frequency_layout.addLayout(type_keys_freq_layout)
        
        frequency_group.setLayout(frequency_layout)
        main_layout.addWidget(frequency_group)
        
        # Alt+Tab settings group
        alt_tab_group = QGroupBox("Alt+Tab Settings")
        alt_tab_layout = QHBoxLayout()
        
        alt_tab_layout.addWidget(QLabel("Tab Presses:"))
        alt_tab_layout.addWidget(QLabel("Min:"))
        self.alt_tab_min_spin = QSpinBox()
        self.alt_tab_min_spin.setRange(1, 10)
        self.alt_tab_min_spin.setValue(self.config['alt_tab_min'])
        alt_tab_layout.addWidget(self.alt_tab_min_spin)
        
        alt_tab_layout.addWidget(QLabel("Max:"))
        self.alt_tab_max_spin = QSpinBox()
        self.alt_tab_max_spin.setRange(1, 20)
        self.alt_tab_max_spin.setValue(self.config['alt_tab_max'])
        alt_tab_layout.addWidget(self.alt_tab_max_spin)
        
        alt_tab_group.setLayout(alt_tab_layout)
        main_layout.addWidget(alt_tab_group)
        
        # Typing options group
        typing_group = QGroupBox("Typing Options")
        typing_layout = QVBoxLayout()
        
        self.random_chars_radio = QRadioButton("Type Random Characters")
        self.random_chars_radio.setChecked(not self.config['use_python_snippets'])
        typing_layout.addWidget(self.random_chars_radio)
        
        self.python_snippets_radio = QRadioButton("Type Python Code Snippets")
        self.python_snippets_radio.setChecked(self.config['use_python_snippets'])
        typing_layout.addWidget(self.python_snippets_radio)
        
        typing_group.setLayout(typing_layout)
        main_layout.addWidget(typing_group)
        
        # Delay settings group
        delay_group = QGroupBox("Delay Settings (seconds)")
        delay_layout = QVBoxLayout()
        
        min_delay_layout = QHBoxLayout()
        min_delay_layout.addWidget(QLabel("Minimum Delay:"))
        self.min_delay_spin = QSpinBox()
        self.min_delay_spin.setRange(1, 60)
        self.min_delay_spin.setValue(self.config['min_delay'])
        min_delay_layout.addWidget(self.min_delay_spin)
        delay_layout.addLayout(min_delay_layout)
        
        max_delay_layout = QHBoxLayout()
        max_delay_layout.addWidget(QLabel("Maximum Delay:"))
        self.max_delay_spin = QSpinBox()
        self.max_delay_spin.setRange(1, 120)
        self.max_delay_spin.setValue(self.config['max_delay'])
        max_delay_layout.addWidget(self.max_delay_spin)
        delay_layout.addLayout(max_delay_layout)
        
        delay_group.setLayout(delay_layout)
        main_layout.addWidget(delay_group)
        
        runtime_group = QGroupBox("Runtime Limit")
        runtime_layout = QHBoxLayout()
        
        self.limit_runtime_checkbox = QCheckBox("Limit Runtime:")
        self.limit_runtime_checkbox.setChecked(False)
        runtime_layout.addWidget(self.limit_runtime_checkbox)
        
        runtime_layout.addWidget(QLabel("Hours:"))
        self.hours_spin = QSpinBox()
        self.hours_spin.setRange(0, 24)
        self.hours_spin.setValue(1)
        runtime_layout.addWidget(self.hours_spin)
        
        runtime_layout.addWidget(QLabel("Minutes:"))
        self.minutes_spin = QSpinBox()
        self.minutes_spin.setRange(0, 59)
        self.minutes_spin.setValue(0)
        runtime_layout.addWidget(self.minutes_spin)
        
        runtime_layout.addWidget(QLabel("Seconds:"))
        self.seconds_spin = QSpinBox()
        self.seconds_spin.setRange(0, 59)
        self.seconds_spin.setValue(0)
        runtime_layout.addWidget(self.seconds_spin)
        
        runtime_group.setLayout(runtime_layout)
        main_layout.addWidget(runtime_group)
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_simulation)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        button_layout.addWidget(self.stop_button)
        
        self.minimize_button = QPushButton("Minimize to Tray")
        self.minimize_button.clicked.connect(self.hide)
        button_layout.addWidget(self.minimize_button)
        
        main_layout.addLayout(button_layout)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Set up the timer for progress bar
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_progress)
        self.current_progress = 0
        self.time_between_actions = 0

        # Set up the timer for elapsed time display 
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)

    def update_elapsed_time(self):
        """Update the elapsed time display and check if runtime limit is reached"""
        if self.running:
            self.elapsed_seconds += 1
            hours, remainder = divmod(self.elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.timer_value.setText(time_str)
            
            # Check if runtime limit is reached
            if self.use_runtime_limit and self.elapsed_seconds >= self.target_runtime:
                # Use QTimer to call stop_simulation in the main thread
                QtCore.QTimer.singleShot(0, self.stop_simulation)
                self.tray_icon.showMessage(
                    "Mouse Activity Simulator", 
                    "Simulation stopped: Runtime limit reached", 
                    QSystemTrayIcon.Information, 
                    2000
                ) 

    def setup_tray(self):
        
        # Create the system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        
        # Load the icon
        app_icon = QIcon("src\\Icon.ico")
        self.tray_icon.setIcon(app_icon)
        # Create the menu
        self.menu = QMenu()
        
        # Add menu actions
        self.show_action = QAction("Show Window", self)
        self.show_action.triggered.connect(self.show)
        self.menu.addAction(self.show_action)
        
        self.start_action = QAction("Start", self)
        self.start_action.triggered.connect(self.start_simulation)
        self.menu.addAction(self.start_action)
        
        self.stop_action = QAction("Stop", self)
        self.stop_action.triggered.connect(self.stop_simulation)
        self.stop_action.setEnabled(False)
        self.menu.addAction(self.stop_action)
        
        self.menu.addSeparator()
        
        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.exit_application)
        self.menu.addAction(self.exit_action)
        
        # Set the menu
        self.tray_icon.setContextMenu(self.menu)
        self.tray_icon.setToolTip("Mouse Activity Simulator by Ahmed Hmila")
        self.tray_icon.show()
        
        # Connect double-click to show window
        self.tray_icon.activated.connect(self.tray_icon_activated)
    
    def tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self.show()
            self.activateWindow()
    
    def apply_dark_theme(self):
        app = QApplication.instance()
        app.setStyle(QStyleFactory.create("Fusion"))
        
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        app.setPalette(dark_palette)
    
    def get_icon(self):
        # Load the icon
        app_icon = QIcon("src\\Icon.ico")
        return app_icon
    
    def update_progress(self):
        if self.running:
            self.current_progress += 1
            value = int((self.current_progress / self.time_between_actions) * 100)
            self.progress_bar.setValue(min(value, 100))
    
    def start_simulation(self):
        # Update configuration from UI
        self.config['min_delay'] = self.min_delay_spin.value()
        self.config['max_delay'] = self.max_delay_spin.value()
        self.config['move_mouse'] = self.move_checkbox.isChecked()
        self.config['scroll_screen'] = self.scroll_checkbox.isChecked()
        self.config['alt_tab'] = self.alt_tab_checkbox.isChecked()
        self.config['type_keys'] = self.type_keys_checkbox.isChecked()
        self.config['alt_tab_min'] = self.alt_tab_min_spin.value()
        self.config['alt_tab_max'] = self.alt_tab_max_spin.value()
        self.config['move_weight'] = self.move_freq_slider.value()
        self.config['scroll_weight'] = self.scroll_freq_slider.value()
        self.config['alt_tab_weight'] = self.alt_tab_freq_slider.value()
        self.config['type_keys_weight'] = self.type_keys_freq_slider.value()
        self.config['use_python_snippets'] = self.python_snippets_radio.isChecked()
        # Reset and start the elapsed time timer
        self.elapsed_seconds = 0
        self.timer_value.setText("00:00:00")
        self.elapsed_timer.start(1000)
        # Check if at least one action is enabled
        if not any([
            self.config['move_mouse'], 
            self.config['scroll_screen'], 
            self.config['alt_tab'],
            self.config['type_keys']
        ]):
            self.tray_icon.showMessage(
                "Error", 
                "Please enable at least one action", 
                QSystemTrayIcon.Warning, 
                2000
            )
            return
        
        self.running = True
        self.start_button.setEnabled(False)
        self.start_action.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_action.setEnabled(True)
        # Calculate target runtime in seconds if limit is enabled
        self.use_runtime_limit = self.limit_runtime_checkbox.isChecked()
        if self.use_runtime_limit:
            hours = self.hours_spin.value()
            minutes = self.minutes_spin.value()
            seconds = self.seconds_spin.value()
            self.target_runtime = hours * 3600 + minutes * 60 + seconds
            if self.target_runtime <= 0:
                self.tray_icon.showMessage(
                    "Error", 
                    "Please set a runtime greater than 0", 
                    QSystemTrayIcon.Warning, 
                    2000
                )
                return
        else:
            self.target_runtime = 0
            # Update UI
            self.status_value.setText("Running")
            self.status_value.setStyleSheet("color: green; font-weight: bold;")
            
        # Get next action immediately to display
        self.choose_next_action()
        
        # Start the automation in a separate thread
        self.automation_thread = threading.Thread(target=self.run_automation)
        self.automation_thread.daemon = True
        self.automation_thread.start()
        
        # Start the progress bar update timer
        self.time_between_actions = self.config['min_delay']
        self.current_progress = 0
        self.update_timer.start(1000)  # Update every second
        
        self.tray_icon.showMessage(
            "Mouse Activity Simulator by Ahmed Hmila", 
            "Activity simulation started", 
            QSystemTrayIcon.Information, 
            2000
        )
    
    def choose_next_action(self):
        """Choose the next action using weighted random selection"""
        available_actions = []
        weights = []
        
        if self.config['move_mouse']:
            available_actions.append('move')
            weights.append(self.config['move_weight'])
            
        if self.config['scroll_screen']:
            available_actions.append('scroll')
            weights.append(self.config['scroll_weight'])
            
        if self.config['alt_tab']:
            available_actions.append('alt_tab')
            weights.append(self.config['alt_tab_weight'])
            
        if self.config['type_keys']:
            available_actions.append('type_keys')
            weights.append(self.config['type_keys_weight'])
        
        if available_actions:
            # Use weighted random choice
            self.next_action_name = random.choices(available_actions, weights=weights, k=1)[0]
            # Update UI with the next action
            action_name = self.next_action_name.replace('_', ' ').title()
            QtCore.QMetaObject.invokeMethod(
                self.next_action_label,
                "setText",
                Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Next action: {action_name}")
            )
        else:
            self.next_action_name = "None"
    
    def stop_simulation(self):
        self.running = False
        self.start_button.setEnabled(True)
        self.start_action.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.elapsed_timer.stop()
        # Update UI
        self.status_value.setText("Stopped")
        self.status_value.setStyleSheet("color: red; font-weight: bold;")
        self.next_action_label.setText("Waiting to start...")
        self.progress_bar.setValue(0)
        
        # Stop the timer
        self.update_timer.stop()
        
        self.tray_icon.showMessage(
            "Mouse Activity Simulator by Ahmed Hmila", 
            "Activity simulation stopped", 
            QSystemTrayIcon.Information, 
            2000
        )
    
    def exit_application(self):
        self.running = False
        self.elapsed_timer.stop()
        # Wait for the thread to finish if it's running
        if hasattr(self, 'automation_thread') and self.automation_thread.is_alive():
            self.automation_thread.join(1)  # Wait for 1 second max
        QApplication.quit()
    
    def closeEvent(self, event):
        # Override close event to minimize to tray instead of closing
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "Mouse Activity Simulator by Ahmed Hmila",
            "Application minimized to tray",
            QSystemTrayIcon.Information,
            2000
        )

    def run_automation(self):
        """Main automation loop that runs in a separate thread"""
        while self.running:
            try:
                # Execute the next action that was chosen
                current_action = self.next_action_name
                
                if current_action == 'move':
                    # Get screen dimensions
                    screen_width, screen_height = pyautogui.size()
                    
                    # Generate random coordinates within screen bounds
                    x = random.randint(0, screen_width - 1)
                    y = random.randint(0, screen_height - 1)
                    
                    # Move mouse to random position with random duration
                    pyautogui.moveTo(x, y, duration=random.uniform(0.5, 1.0))
                    
                elif current_action == 'scroll':
                    # Scroll a random amount
                    scroll_amount = random.randint(-400, 200)
                    pyautogui.scroll(scroll_amount)
                    
                elif current_action == 'alt_tab':
                    # Press Alt+Tab to switch windows
                    keyboard.press('alt')
                    
                    # Press tab a random number of times (configured range)
                    tab_presses = random.randint(
                        self.config['alt_tab_min'], 
                        self.config['alt_tab_max']
                    )
                    for _ in range(tab_presses):
                        keyboard.press('tab')
                        time.sleep(0.1)
                        keyboard.release('tab')
                    
                    time.sleep(0.5)
                    keyboard.release('alt')

                elif current_action == 'type_keys':
                    if self.config['use_python_snippets']:
                        # Type a Python code snippet
                        keyboard.press('tab')
                        time.sleep(0.1)
                        keyboard.press('enter')
                        snippet = random.choice(self.python_code_snippets)
                        pyautogui.write(snippet, interval=0.05)
                        # Add newline at the end
                        pyautogui.press('enter')

                    else:
                        # Type random characters
                        safe_chars = string.ascii_lowercase + string.digits + ' '
                        num_keys = random.randint(3, 8)
                        
                        # Type the random keys with slight delay between each
                        for _ in range(num_keys):
                            key = random.choice(safe_chars)
                            pyautogui.write(key, interval=0.1)
                
                # Choose the next action for display
                self.choose_next_action()
                
                # Wait for a random interval between actions
                delay = random.uniform(self.config['min_delay'], self.config['max_delay'])
                
                # Update the progress bar max time
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "set_time_between_actions",
                    Qt.QueuedConnection,
                    QtCore.Q_ARG(int, int(delay))
                )
                
                # Reset progress counter
                self.current_progress = 0
                
                time.sleep(delay)
                
            except Exception as e:
                print(f"Error during automation: {e}")
                time.sleep(5)  # Wait before retrying
    
    @QtCore.pyqtSlot(int)
    def set_time_between_actions(self, seconds):
        self.time_between_actions = seconds
        self.current_progress = 0
        self.progress_bar.setValue(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep the app running when the window is closed
    app_icon = QIcon("src\\Icon.ico")
    app.setWindowIcon(app_icon)
    window = MouseActivitySimulator()
    window.show()  # Show the window on startup
    
    sys.exit(app.exec_())