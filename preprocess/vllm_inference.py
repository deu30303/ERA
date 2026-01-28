import torch
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import string  # [핵심 수정] 이 줄이 빠져서 에러가 났습니다.
import re      # re.sub 사용을 위해 필요
from collections import Counter # Counter 사용을 위해 필요



def extract_answer(args, text):
    if not text:
        return ""


    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


    pattern = re.compile(r'Answer:\s*', re.IGNORECASE)
    parts = pattern.split(text)
    if len(parts) > 1:
        text = parts[-1] 


    stop_phrases = [
        '\nQuestion', 'Question:', '.Question', 
        '\nContext', 'Context:', '.Context',    
        '\nExample', 'Example:',                
        '\n\n'                                 
    ]
    
    for phrase in stop_phrases:
        if phrase in text:
            text = text.split(phrase)[0]

    text = text.strip()
    

    while text and (text.endswith('.') or text.endswith('\n')):
        text = text[:-1].strip()


    if ',' in text:
        text_comma = text.split(',')
        if len(text_comma) > 2: 
            text = text_comma[0].strip()

    return text



from transformers import AutoTokenizer
from vllm import LLM, SamplingParams



def vllm_wo_retrieval(args, data):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    stop_token_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for stop_str in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]:
            try:
                stop_id = tokenizer.convert_tokens_to_ids(stop_str)
                if isinstance(stop_id, int) and stop_id < tokenizer.vocab_size:
                    stop_token_ids.append(stop_id)
            except:
                pass
    stop_token_ids = list(set(stop_token_ids))

    rep_penalty = 1
    

    sys = ''
    examples = []

    prompt_template = '{system_prompt}\n\nDirectly answer the question without any other words.{query}'

    if 'Llama' in args.model_name_or_path or 'Qwen' in args.model_name_or_path or 'Mistral' in args.model_name_or_path:
        rep_penalty = 1.1
        sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
        examples = [
            '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
            '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
            '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
        ]

        prompt_template = '{system_prompt}\n\nDirectly answer the question without any other words.{query}'


    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1, tokenizer_mode='auto',
              trust_remote_code=True, load_format='auto', gpu_memory_utilization=0.95, 
              max_num_batched_tokens=16384)


    sampling_param = SamplingParams(
                n=args.infer_k,
                max_tokens=512,
                top_k=50,
                top_p=0.6,
                temperature=0.3,
                repetition_penalty=rep_penalty,
                stop_token_ids=stop_token_ids) 

    sens = []
    

    for sample in data:
        full_examples = "".join(examples)
        query_str = '\nQuestion: ' + sample['question'] + '\nAnswer:'
        sentence = prompt_template.format(
            system_prompt=sys + full_examples, 
            query=query_str
        )

        sens.append(sentence)

    outputs = llm.generate(sens, sampling_params=sampling_param)
    
    ret = []
    for output in outputs:
        generated_k_text = []
        for i in range(args.infer_k):
            generated_text = output.outputs[i].text
            generated_k_text.append(generated_text)
        ret.append(generated_k_text)

    return ret







def vllm_w_retrieval(args, data):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    stop_token_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for stop_str in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
            try:
                stop_id = tokenizer.convert_tokens_to_ids(stop_str)
                if isinstance(stop_id, int) and stop_id < tokenizer.vocab_size:
                    stop_token_ids.append(stop_id)
            except:
                pass
    stop_token_ids = list(set(stop_token_ids))

    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1, tokenizer_mode='auto',
              trust_remote_code=True, load_format='auto', gpu_memory_utilization=0.95, 
              max_num_batched_tokens=16384)

    sampling_param = SamplingParams(
        n=args.infer_k,
        max_tokens=512,
        top_k=50,
        top_p=0.6,
        temperature=0.3,
        repetition_penalty=1.1,
        stop_token_ids=stop_token_ids 
    )
    


    sys_instruction = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences.'
    examples_str = ""
    
    if 'Llama' in args.model_name_or_path or 'Qwen' in args.model_name_or_path or 'Mistral' in args.model_name_or_path:
        examples_str = (
            "Example 1:\nQuestion: What is the capital of France?\nAnswer: Paris.\n\n"
            "Example 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.\n\n"
            "Example 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.\n\n"
        )

    sens = []
    for sample in data:
        if 'ctxs' in sample:
            ctxs_list = sample['ctxs']
        elif 'possible_golden_ctxs' in sample:
            ctxs_list = sample['possible_golden_ctxs'] + sample.get('possible_noisy_ctxs', [])
        else:
            ctxs_list = []

        ctxs_text = ''
        for i in range(min(3, len(ctxs_list))):
            ctxs_text += f'\nContext{i+1}: {ctxs_list[i]["text"]}'


        header = f"{sys_instruction}\n\n{examples_str}"
        

        body = f"The following contexts will help you complete the question-and-answer pair.{ctxs_text}\n\nQuestion: {sample['question']}\nAnswer:"
        

        prompt_str = header + body
        
        sens.append(prompt_str)

    outputs = llm.generate(sens, sampling_params=sampling_param)
    
    ret = []
    for output in outputs:
        generated_k_text = []
        for i in range(args.infer_k):
            generated_text = output.outputs[i].text
            generated_k_text.append(generated_text)
        ret.append(generated_k_text)

    return ret



def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
