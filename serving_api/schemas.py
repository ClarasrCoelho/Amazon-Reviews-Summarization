from pydantic import BaseModel, Field
from typing import List, Optional, Annotated

class ReviewCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    rating: Annotated[float, Field(ge=1, le=5)]
    user: str = "local-user"


class ReviewIn(BaseModel):
    text: str
    rating: float | None = None
    user: str = "local-user"


class BookSummary(BaseModel):
    asin: str
    title: str
    summary: str
    avg_sentiment: float
    avg_aspect_polarity: float
    aspects: List[str]
