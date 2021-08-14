import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseSettings
from cachetools import LRUCache

from data.db import DbConnection


class Settings(BaseSettings):
    run: str
    items_per_cluster: int = 50


app = FastAPI()
settings = Settings()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

db = DbConnection(settings.run)
cache = LRUCache(maxsize=100)


def get_cluster_data(lemma: str):
    if lemma not in cache:
        cache[lemma] = db.get_clusters_for_lemma(lemma)

    return cache[lemma]


@app.get("/search", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})


@app.get("/clusters")
async def cluster_data(word: str):

    return {'clusters': [
        {
            'name': 'First Cluster',
            'data': [
                        {'x': 1, 'y': 1, 'z': 0, 'text': 'I like [dogs].'}
                    ]
        },
        {
            'name': 'Second Cluster',
            'data': [
                {'x': -1, 'y': -2, 'z': 1, 'text': 'He was a [dog].'}
            ]
        },
    ]}
