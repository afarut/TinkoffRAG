from decouple import config


DEFAULT_SYSTEM_PROMPT = ""
TOKEN = config('TOKEN', default="", cast=str)
if TOKEN == "":
	raise Exception("Укажите токен из huggingface")