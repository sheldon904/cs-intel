"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    decision: str = Field(..., description="'approved' or 'rejected'")
    note: str = Field("", description="optional operator note")


class ProposeResponse(BaseModel):
    proposal: dict
    memory: dict


class DecisionResponse(BaseModel):
    proposal: dict
    memory: dict
