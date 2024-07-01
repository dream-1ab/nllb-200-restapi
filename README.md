# gradio based simple app and ready-to-use web api with FastApi for [https://github.com/facebookresearch/fairseq]()

### Screenshots:

* Web api:
![](./resources/Screenshot_20240701_083523.png)
![](./resources/Screenshot_20240701_092731.png)
![](./resources/Screenshot_20240701_093105.png)
* Web app
![](./resources/Screenshot_20240701_092605.png)

This api and example app uses CUDA if available by default.
there was a wrapper class around huggingface official api named ```my_translator.py``` so you can easly integrate it into your own projects.

## How to use it?
it is very simple to use, by default this uses CUDA for the computation backend if available otherwise uses CPU.
* install the necessary python packages through `pip3 install -r ./requirements.py`
* start the app with ```python3 ./translator_app.py``` or rest api with ```python3 ./translator_api.py```

## How to specify which models we want wanna use?

in order to use the wrapper you need to initialize the translator class using ```my_translator = Translator("1.3b")```
the literal ```1.3b``` is the simplified model name and all the available values are:  ```"600m", "1.3b", "3.3b", "54b"``` or model name from hugging face e.g. ```facebook/nllb-200-distilled-600M``` or anything else, for further details: please see the source code: ```my_translator.py```