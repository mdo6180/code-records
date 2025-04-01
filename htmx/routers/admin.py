from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter()

@router.get("/hello", response_class=HTMLResponse)
async def hello() -> str:
    return """
    <div>hello there</div>
    """