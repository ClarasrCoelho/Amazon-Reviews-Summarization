from pydantic import BaseModel, Field
from typing import List, Optional


class ReviewIn(BaseModel):
    asin: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=4000)
    rating: float | None = None
    user: str | None = "local-user"

class BookSummary(BaseModel):
    asin: str
    title: str
    summary: str
    avg_sentiment: float
    avg_aspect_polarity: float
    aspects: List[str]
