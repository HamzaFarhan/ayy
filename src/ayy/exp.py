from pydantic import BaseModel, Field
from pathlib import Path


class Animal(BaseModel):
    name: str = Field(description="The name of the animal")
    age: int = Field(description="The age of the animal")
    sound: str = Field(description="The sound the animal makes")


class Dog(Animal):
    breed: str = Field(description="The breed of the dog")


