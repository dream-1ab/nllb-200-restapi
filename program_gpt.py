
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, NllbTokenizer
import ollama
import my_translator
import json

translator = my_translator.Translator("1.3b")
translator.set_direction("uig_Arab", "eng_Latn")

messages: list[ollama.Message] = [
    {
        "role": "system",
        # "content": translator.translate("سىز بىر سۈنئىي ئەقىل ئائىلە ياردەمچىسى، سىز چىراقلارنى ئۆچۈرۈپ ياندۇرالايسىز ۋە چىراقنىڭ رەڭگىنى تەڭشىيەلەيسىز، پەقەت ئىشلەتكۈچى سورىغان سوئالغا ئىخچام جاۋاب بېرىڭ ۋە بەك كۆپ گەپ قىلماڭ")
        "content": "You are a smart home device assistant, answer users question as simple as possible, no more description and additional emotions, special characters and markdown formats."
    },
    {
        "role": "system",
        # "content": translator.translate("سىز بىر سۈنئىي ئەقىل ئائىلە ياردەمچىسى، سىز چىراقلارنى ئۆچۈرۈپ ياندۇرالايسىز ۋە چىراقنىڭ رەڭگىنى تەڭشىيەلەيسىز، پەقەت ئىشلەتكۈچى سورىغان سوئالغا ئىخچام جاۋاب بېرىڭ ۋە بەك كۆپ گەپ قىلماڭ")
        "content": "You have access to the following features:\nControl home lamp (e.g. change the lamp color, turn it on/off) and control the electric fan (fan speed), and also you can simply remember persons name."
    }
]
print("Ready")

while True:
    text = input()
    if text == "dump_message":
        print(json.dumps(messages))
        continue
    translated = translator.translate(text)
    print(f"translated: {translated}")
    messages.append({
        "role": "user",
        "content": translated
    })
    response = ollama.chat(messages=messages, stream=False, model="llama3")
    translator.swap_direction()
    result = translator.translate(response["message"]["content"])
    translator.swap_direction()
    messages.append(response["message"])
    print("\n")
    print(result)
    print(f"translated: {response['message']['content']}")
    print("--------------------------------\n\n")
    