from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseSettings, BaseModel
from web_helper import compute_search_data, ClusterSearchData, subcluster_search


class Settings(BaseSettings):
    run: str
    items_per_cluster: int = 50


app = FastAPI()
settings = Settings()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class SearchRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
async def cluster_data(request: SearchRequest):
    return compute_search_data(request.text)


@app.post('/subcluster')
async def subcluster(request: ClusterSearchData):
    return subcluster_search(request)


