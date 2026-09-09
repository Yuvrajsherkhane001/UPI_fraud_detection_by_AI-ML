 # api/index.py
  from fastapi import FastAPI
  from src.api import app as fastapi_app

  # Export the FastAPI app for Vercel
  app = fastapi_app
