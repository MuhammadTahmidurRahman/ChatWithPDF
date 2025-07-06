import os
import shutil
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pdf_chat import build_chain, ask_chain

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=HTMLResponse)
async def query(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...)
):
    # 1) save the uploaded PDF
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) build your chain and run it
    chain = build_chain(file_path)
    answer = ask_chain(chain, question)

    # 3) render result page
    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": file.filename,
        "question": question,
        "answer": answer,
    })
