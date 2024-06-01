import pandas as pd
from config import TOKEN, HyperParameters, DEFAULT_SYSTEM_PROMPT
from huggingface_hub.hf_api import HfFolder
import warnings
import json
from transformers import logging
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

HfFolder.save_token(TOKEN)
logging.set_verbosity_error() # set_verbosity_warning()
warnings.filterwarnings('ignore')


base_dir = './'
text_temp = '''{title}
{description}
'''

art_df = pd.DataFrame(json.load(open(f'{base_dir}/dataset.json')))
art_df = art_df[art_df['description'].str.len() > 5].reset_index(drop=True)


qdrant_client = QdrantClient('path', port=6333, timeout=60)


def vec_search(bi_encoder, ENCODER_NAME, query, n_top_cos):
    COLL_NAME = ENCODER_NAME.replace('/','_')
    
    # Кодируем запрос в вектор
    query_emb = str_to_vec(bi_encoder, query)

    # Поиск в БД
    search_result = qdrant_client.search(
        collection_name = COLL_NAME,
        query_vector = query_emb,
        limit = HyperParameters["n_top_cos"],
        with_vectors = False
    )
    
    top_art = [x.payload['description'] for x in search_result]
    top_url = [x.payload['url'] for x in search_result]
    top_title = [x.payload['title'] for x in search_result]
    
    return top_title, top_art, top_url



def get_bi_encoder(bi_ENCODER_NAME):
    raw_model = Transformer(model_name_or_path=f'{bi_ENCODER_NAME}')
    
    bi_encoder_dim = raw_model.get_word_embedding_dimension()
    
    pooling_model = Pooling(
        bi_encoder_dim,
        pooling_mode_cls_token = False,
        pooling_mode_mean_tokens = True
    )
    bi_encoder = SentenceTransformer(
        modules = [raw_model, pooling_model],
        device = DEVICE # cuda DEVICE
    )
    
    return bi_encoder, bi_encoder_dim

# Формируем из строки вектор
def str_to_vec(bi_encoder, text):
    embeddings = bi_encoder.encode(
        text,
        convert_to_tensor = True,
        show_progress_bar = False
    )
    return embeddings




def get_llm_answer(query, chunks_join, top_p, model, tokenizer):

    prompt = DEFAULT_SYSTEM_PROMPT.format(chunks_join=chunks_join, query=query)

    def generate(model, tokenizer, prompt):
        data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(model.device) for k, v in data.items()}
        output_ids = model.generate(
            **data,
            bos_token_id=128000,
            eos_token_id=128009,
            pad_token_id=128000,
            do_sample=True,
            max_new_tokens=HyperParameters['max_new_tokens'],
            no_repeat_ngram_size=15,
            repetition_penalty=1.12,
            temperature=HyperParameters['temperature'],
            top_k=HyperParameters['top_k'],
            top_p=HyperParameters['top_p'] 
        )[0]
        output_ids = output_ids[len(data["input_ids"][0]) :]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        return output.strip()

    response = generate(model, tokenizer, prompt)
    
    return response


result = []

def get_hyeat(query, model, tokenizer):
    bi_encoder, vec_size = get_bi_encoder(ENCODER_NAME)

    top_title, top_art, top_url = vec_search(bi_encoder, ENCODER_NAME, query, HyperParameters["n_top_cos"])

    chunks = [text_temp.format(title=t, description=a) for t, a in zip(top_title, top_art)]
    chunks_join = HyperParameters['join_sym'].join(chunks)

    answer = get_llm_answer(query, chunks_join, top_p, model, tokenizer)
    return answer