"""Pydantic models for content.json validation."""

from typing import Optional
from pydantic import BaseModel, Field


class Links(BaseModel):
    """Project or resource links."""
    live: Optional[str] = None
    github: Optional[str] = None


class DateRange(BaseModel):
    """Date range for experience, education, or projects."""
    start: str
    end: Optional[str] = None


class ProjectOutcomes(BaseModel):
    """Project outcomes and impact."""
    metrics: Optional[str] = None
    impact: Optional[str] = None


class Project(BaseModel):
    """Individual project."""
    title: str
    tagline: str
    description: str
    problem: Optional[str] = None
    tech_stack: list[str]
    architecture: Optional[str] = None
    features: list[str] = []
    challenges: Optional[str] = None
    outcomes: Optional[ProjectOutcomes] = None
    links: Optional[Links] = None
    tags: list[str] = []


class Experience(BaseModel):
    """Work experience entry."""
    id: str
    company: str
    position: str
    location: Optional[str] = None
    type: str
    date_range: DateRange
    description: str
    responsibilities: list[str] = []
    achievements: list[str] = []
    tech_stack: list[str] = []
    projects_worked_on: list[str] = []
    tags: list[str] = []


class Skills(BaseModel):
    """Skills - simple string format."""
    languages: str
    frameworks: str
    tools_and_platforms: str


class Education(BaseModel):
    """Education entry."""
    institution: str


class Bio(BaseModel):
    """Personal bio and contact info."""
    name: str
    title: str
    summary: str
    location: Optional[str] = None
    availability: Optional[str] = None
    contact_preferences: Optional[str] = None


class Metadata(BaseModel):
    """Content metadata."""
    last_updated: str


class Content(BaseModel):
    """Complete content.json structure."""
    metadata: Metadata
    bio: Bio
    projects: list[Project] = []
    experience: list[Experience] = []
    skills: Optional[Skills] = None
    education: list[Education] = []


# Helper function to load and validate
def load_content(filepath: str = "contents.json") -> Content:
    """Load and validate content.json file."""
    import json
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return Content(**data)