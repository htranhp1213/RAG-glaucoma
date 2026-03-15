from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_ROOTS = [
    Path("./RAG-glaucoma/PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0/FundusImages").resolve(),
    Path(".").resolve(),
]


class AskRequest(BaseModel):
    prompt: str


class AskResponse(BaseModel):
    answer: str
    image_url: Optional[str] = None


def is_allowed_image_path(file_path: Path) -> bool:
    file_path = file_path.resolve()
    for root in ALLOWED_IMAGE_ROOTS:
        try:
            file_path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


@app.get("/")
def home():
    return {"message": "RAG backend is running"}


@app.get("/image")
def get_image(path: str = Query(...)):
    file_path = Path(path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found.")

    if not is_allowed_image_path(file_path):
        raise HTTPException(status_code=403, detail="Image path not allowed.")

    return FileResponse(file_path)


@app.post("/ask", response_model=AskResponse)
def ask_backend(data: AskRequest):
    prompt_text = data.prompt.strip()

    if not prompt_text:
        return AskResponse(
            answer="Please enter a question.",
            image_url=None,
        )

    from prompt import ask_question, RAGConfig

    cfg = RAGConfig(
        llm_model="google/flan-t5-base",
        max_new_tokens=200,
        text_k=3,
        image_k=2,
        top_k=5,
    )

    result = ask_question(prompt_text, cfg)

    answer = result.get("answer", "No answer returned.")
    image_url = None

    recommended_images = result.get("recommended_images", [])
    if recommended_images:
        first_image = recommended_images[0]
        file_path = first_image.get("file_path")

        if file_path:
            image_url = f"http://127.0.0.1:8000/image?path={quote(file_path)}"

    return AskResponse(
        answer=answer,
        image_url=image_url,
    )
