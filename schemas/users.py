from pydantic import BaseModel

# Properties to receive via API on user creation
class UserCreate(BaseModel):
    email: str
    password: str

# Properties to return via API, hiding the password
class User(BaseModel):
    id: int
    email: str
    

    class Config:
        from_attributes = True
