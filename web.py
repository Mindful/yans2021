from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


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
