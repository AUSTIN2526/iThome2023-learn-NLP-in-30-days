import os
import openai
import pandas as pd

from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
load_dotenv()


    
def load_simple(path):
    df = pd.read_csv(path)
    Q = df['question']
    A = df['answer']
    
    
    return [f'question:{q} answer:{a}' for q, a in zip(Q, A)]

    
def creat_fewshot(model, inputs, simple, num = 10):
    simple_emb = model.encode(simple)
    inputs_emb = model.encode([inputs])
    
    cos_sim = util.cos_sim(simple_emb, inputs_emb)
    combo = [[cos_sim[i], i] for i in range(len(cos_sim))]
    combo = sorted(combo, key=lambda x: x[0], reverse=True)
    
    few_shot = [simple[i] for _, i in combo[:num]]
    
    return few_shot
    
def gpt_instruct(dialog, few_shot):
    instruct = '你是客服人員，在接收到用戶詢問時，需要盡可能地以專業、簡短且易懂的方式提供答案。如果用戶的問題不夠清晰，你需要引導他們提供更多訊息以便更準確地回答。以下是一些你可能需要參考的資料，以便更有效地應對用戶的問題。'
    init_instruct = instruct + '\n' + "".join(few_shot)
    
    return {"role": "system", "content": f'{init_instruct}'}
    
    

def GPT(model, dialog, simple, gpt_version, TYPE, num = 10):
    few_shot = creat_fewshot(model, dialog[-1], simple, num)
    dialog[0] = gpt_instruct(dialog, few_shot)
    if TYPE == 'azure':
        response = openai.ChatCompletion.create(
            engine=gpt_version, 
            messages=dialog
        )
    else:
        response = openai.ChatCompletion.create(
            model=gpt_version,
            messages=dialog
        )  
        
    return response.choices[0].message.content
    
  

if os.getenv('API_TYPE') != 'azure':
    openai.api_key = os.getenv('API_KEY')
else:
    openai.api_type = os.getenv('API_TYPE')
    openai.api_version = os.getenv('API_VERSION')
    openai.api_base = os.getenv('API_ENDPOINT')
    openai.api_key = os.getenv('API_KEY')
    gpt_version = os.getenv('GPT_VERSION')



simple = load_simple('qa_data.csv')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
dialog = [[]]
while(1):
    user_input = input('請輸入問題:')
    dialog.append({"role": "user", "content": f'{user_input}'})
    response = GPT(model, dialog, simple, gpt_version, os.getenv('API_TYPE'))
    dialog.append({"role": "assistant", "content": f'{user_input}'})
    print('GPT回復:',response)

    


