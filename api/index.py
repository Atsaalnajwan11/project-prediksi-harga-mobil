"""
=============================================================
Vercel Serverless Entry Point
=============================================================
Vercel untuk Python wajib pakai folder api/.
File ini import app dari app.py agar bisa di-serve sebagai serverless function.
"""
import sys
import os

# Tambahkan parent directory ke path agar bisa import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Vercel Python runtime otomatis mendeteksi `app` sebagai WSGI application