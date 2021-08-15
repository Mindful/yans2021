import asyncio
from itertools import cycle
from typing import Tuple

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseSettings
from spacy.tokens import Token

from data.db import DbConnection
from nlp.embedding import EmbeddingExtractor
from web_helper import get_data_for_search


class Settings(BaseSettings):
    run: str
    items_per_cluster: int = 50


app = FastAPI()
settings = Settings()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/search")
async def cluster_data(text: str):
    data = get_data_for_search(text)
    if len(data) == 0:
        raise RuntimeError('No data for that word')
    return data
