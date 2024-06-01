from decouple import config


DEBUG=config('DEBUG', default=True, cast=bool)
DEFAULT_SYSTEM_PROMPT = ""
TOKEN = config('TOKEN', default="", cast=str)
if TOKEN == "":
	raise Exception("Укажите токен из huggingface")