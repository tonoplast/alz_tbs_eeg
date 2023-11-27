# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:43:32 2023

This creates a batch file that can be double clicked to run the whole process.

@author: sungw
"""

from pathlib import Path

project_path = Path.home().joinpath('GitHub','tram_protocol_eeg')

myBat = open(rf'{project_path}\run_pac.bat','w+')
myBat.write('''
            @echo off
            setlocal enabledelayedexpansion
            
            REM Change the 'username_new' to the username before the git repository (eg. omurphy10)
			set username_new="wooks"
            
            REM Specify the folder path for the log file
            set "LOG_FOLDER=C:\\Users\\%username_new%\\GitHub\\alz_tbs_eeg\\src\\analysis\\pac\\logs"
            
            REM Get the current date and time in the desired format
            for /f "tokens=2 delims==" %%I in ('wmic OS Get localdatetime /value') do set "dt=%%I"
            set "YYYY=%dt:~0,4%"
            set "MM=%dt:~4,2%"
            set "DD=%dt:~6,2%"
            set "HH=%dt:~8,2%"
            set "Min=%dt:~10,2%"
            set "SS=%dt:~12,2%"
            set "LOG_FILE=%LOG_FOLDER%\\logs.%YYYY%%MM%%DD%-%HH%%Min%%SS%.txt"
            
            REM Activate the Anaconda environment and run the Python script
            call C:\\Users\\%username_new%\\Anaconda3\\Scripts\\activate.bat
            cd C:\\Users\\%username_new%\\GitHub\\alz_tbs_eeg
            call conda activate alz_tbs_eeg
            call python -u -m src.analysis.pac.main | wsl tee "%LOG_FILE%"
            
            pause
            ''')
            
myBat.close()