from decouple import config


DEBUG = config('DEBUG', default=True, cast=bool)

DEFAULT_SYSTEM_PROMPT = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.<|eot_id|><|start_header_id|>user<|end_header_id|>
Используй только следующий контекст, чтобы кратко ответить на вопрос в конце. Если ответа не найден напиши "Ответ не найден". Не пытайся выдумывать ответ.
Контекст:
###
{chunks_join}
###
Вопрос:
###
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

TOKEN = config('TOKEN', default="", cast=str)
if TOKEN == "":
	raise Exception("Укажите токен из huggingface")

ENCODER_NAME = "intfloat/multilingual-e5-large"

MODEL_CONFIG = {
	'join_sym': '\n\n',
	'n_top_cos': 5,
	'top_cos_limit': 0.6469240878510235,
	'max_new_tokens': 1404,
	'temperature': 0.8813610841058304,
	'top_p': 0.42172468879174635,
	'top_k': 146
}
DEVICE = "CUDA"