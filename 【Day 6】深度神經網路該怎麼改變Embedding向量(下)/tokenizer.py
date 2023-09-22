class Tokenizer:
    def __init__(self, english_sentence, max_len = 5, special_token = None, padding = True):
        
        tokens = []
        for sentence in english_sentence:
            tokens.extend(sentence.split(' '))  # 將一段句字進行斷詞後加入列表(List)
        tokens = sorted(set(tokens))            # 通過set()過濾重複單字
        
        if special_token is not None:
            tokens = special_token + list(tokens)
        
        self.token2num = {tokens:num for num, tokens in enumerate(tokens)}
        self.num2token = {num:tokens for num, tokens in enumerate(tokens)}
        
        self.max_len = max_len
        self.padding = padding
    
    def __call__(self, input_text):
        tokens = input_text.split(' ')              
        UNK_IDX = self.token2num['[UNK]']
        PAD_IDX = self.token2num['[PAD]'] 

        output_num = []
        for token in tokens:
            num = self.token2num.get(token, UNK_IDX)  # 轉換成數字(不存在於字典時轉換成UNK_IDX)
            output_num.append(num)
            
        padding_num = self.max_len - len(output_num)  # 計算需填充的數量
        return output_num + [PAD_IDX] * padding_num   # 補齊最大長度
       
    
    def num2tokens(self, input_list):
        output_list = [self.num2token[num] for num in input_list]
        return ' '.join(output_list)
    
    
