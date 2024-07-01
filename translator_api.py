#
 # @author [Dream lab مۇختەرجان مەخمۇت]
 # @email [ug-project@outlook.com]
 # @create date 2024-06-26 15:18:27
 # @modify date 2024-06-26 15:18:27
 # @desc [description]
#

import fastapi
from fastapi import Query, Body, Path
import uvicorn
import my_translator
from pydantic import BaseModel, Field
from typing import TypeVar, Sequence, Generic, Annotated

DATA_TYPE = TypeVar("DATA_TYPE")

class BaseResponse(BaseModel, Generic[DATA_TYPE]):
    message: str
    succeed: bool
    data: DATA_TYPE

class LanguageListResponse(BaseModel):
    data: list[str]

class TranslationResponse(BaseModel):
    source_language: my_translator.Language
    target_language: my_translator.Language
    source_text: str
    translated_text: str

class TranslationRequest(BaseModel):
    source_language: my_translator.Language
    target_language: my_translator.Language
    text: str

my_app = fastapi.FastAPI()
translator = my_translator.Translator("1.3b")
print("Translator model is ready.")

@my_app.get("/api/translator/language_list")
async def language_list() -> BaseResponse[LanguageListResponse]:
    return BaseResponse(message="Succeed", succeed=True, data=LanguageListResponse(data=my_translator.language_list))

direction = ("", "")
@my_app.post("/api/translator/translate")
async def translate(request: TranslationRequest) -> BaseResponse[TranslationResponse]:
    global direction
    (source_language, target_language, text) = (request.source_language, request.target_language, request.text)
    if (source_language, target_language) != direction:
        direction = (source_language, target_language)
        translator.set_direction(source_language, target_language)
    translated = translator.translate(text=text)
    return BaseResponse[TranslationResponse](message="Succeed", succeed=True, data=TranslationResponse(source_language=source_language, target_language=target_language, source_text=text, translated_text=translated))


uvicorn.run(my_app, port=8888)
