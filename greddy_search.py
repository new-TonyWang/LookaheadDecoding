from transformers import GenerationMixin
import torch
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GreedySearchOutput, SampleOutput, TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper
import torch.distributed as dist
import os, time, random
FUNC_MAP = {}
CONFIG_MAP = {}
COLOR_PRINT = int(os.environ.get("COLOR_PRINT", 0))

def jacobi_greedy_search_multilevel(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    synced_gpus: bool = False,
    
    stop_token: Optional[str]= None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    # 初始化一堆参数
    # ......
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor= torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    
    this_peer_finished = False  # used by synced_gpus only
    ############### configurations 
    # W
    WINDOW_SIZE = CONFIG_MAP.get("WINDOW_SIZE", 60)
    # G
    GUESS_SET_SIZE = CONFIG_MAP.get("GUESS_SET_SIZE", 60)
    ALWAYS_FWD_ONE = CONFIG_MAP.get("ALWAYS_FWD_ONE", 1)
    # N
    LEVEL = CONFIG_MAP.get("LEVEL", 8)
    # 是否从prompt中初始化past_tokens
    POOL_FROM_PROMPT = CONFIG_MAP.get("POOL_FROM_PROMPT", 0)

    # N-1
    GUESS_SIZE = LEVEL - 1

    ############### Init methods

    all_old_tokens = input_ids[0].tolist()
    init_len = len(all_old_tokens)

    def copy_from():
        return random.choice(all_old_tokens)

    set_token = copy_from
    
    '''
    正常推理的时候，past_tokens的shape是 (LEVEL-1) * W。
    但是在预热阶段，每一次跑前向都会推出一个正确的token，从而导致lookahead分支的窗口变小，于是在初始化的窗口需要多分配LEVEL - 2个token，
    然后经过N-1次预热之后，窗口的长度就会变成WINDOW_SIZE
    '''
    past_tokens = [[set_token() for _ in range(WINDOW_SIZE + LEVEL - 3)]] + [None for _ in range(LEVEL - 2)]
    #past_tokens is the lookahead window. Current we initialize it with random copy from prompts
    
    ###############end Init methods
    # fill_level指的是预热阶段运行到了第次，最大为LEVEL-1
    fill_level = 0
    guess_tokens = None
    token_map = {}
    steps = 0
    guess_skip_dist = 0

    if POOL_FROM_PROMPT:
        # 先通过prompt预填充token_map
        # 每一个token只保存GUESS_SET_SIZE个可能序列,每一个token后面会猜测LEVEL-1个词
        fill_pool_with_prompt(all_old_tokens, token_map, LEVEL, GUESS_SET_SIZE)

    # 自投机解码
    while True:
        
        # 准备自回归解码的输入
        past_key_values = model_kwargs.pop("past_key_values", None)
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if past_key_values is None:
            model_inputs["input_ids"] = input_ids
        else:
            model_inputs["input_ids"] = model_inputs["input_ids"][:, -1 - guess_skip_dist:]
            model_inputs["position_ids"] = model_inputs["position_ids"][:, -1 - guess_skip_dist:]
        model_inputs["past_key_values"] = past_key_values

        #set up guess_tokens for verification branch 
        # past_tokens[LEVEL - 2] is not None means we are still in warmup stage filling multi-level window
        # lst_token 就是最后一个生成的token，如果past_tokens没有被填满的话，就不会有产生guess_tokens
        if past_tokens[LEVEL - 2] is not None and lst_token in token_map and GUESS_SET_SIZE > 0:  
            guess_tokens_ = token_map[lst_token]
            guess_tokens = []
            for tok in list(guess_tokens_): #一次校验多组猜测
                guess_tokens += list(tok)
        else:
            guess_tokens = None
        # ...
        past_tokens_inp = past_tokens
            
        outputs = self.jforward_multilevel(
            **model_inputs,
            past_tokens=past_tokens_inp,
            guess_tokens=guess_tokens,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            level=LEVEL,
            WINDOWS_SIZE=WINDOW_SIZE,
            guess_size=GUESS_SIZE,
            fill_level=fill_level,
            la_mask_offset=0,
        )
        
        steps += 1
        
        if past_tokens[LEVEL - 2] is None: #预热阶段的输出
            next_token_logits = outputs.out_logits
        else:
            next_token_logits = outputs.out_logits #outputs.logits[:, -1, :]

        # pre-process distribution
        #next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens_scores = next_token_logits
        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        first_guess = next_tokens.item()
        max_hit = 0 
        hits = [first_guess] + [0] * (GUESS_SIZE - 1)

        new_results = []

        if past_tokens[1] is None: #filling multi-level window, the very first step is different
            assert fill_level == 0
            # prefill阶段，每一个token都输出下一个token的logits
            # past_tokens[0] 是第一次从prompt随机填入的，并且删除了第一个token
            # past_tokens[1] 是past_token[0]通过模型输出的token 
            past_tokens[0] = past_tokens[0][1:] 
            past_tokens[1] = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()
            fill_level += 1
        elif past_tokens[LEVEL - 2] is None: #filling multi-level window，
            # 每一次推理都至少有一个是正确的，他们把正确的从past_tokens里面排除
            for level in range(fill_level + 1):
                past_tokens[level] = past_tokens[level][1:] 
            current_past_tokens = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()

            past_tokens[fill_level + 1] = current_past_tokens[1:]


            fill_level += 1
        else: 
            # 预热完毕之后运行的分支
            if guess_tokens is not None:
                # 校验通过n-gram猜测的token是否和输出的guess_logits有相同。
                # 如果有相同则取相同数量最多的输出。例如n-gram猜测的是who->are->you和who->is->he，
                # 输出的是guess_logits是who->are->they，则相当于猜对了who->are两个token
                guess_results = torch.argmax(outputs.guess_logits, dim=-1)[0].tolist()
                for eg in range(len(guess_results) // GUESS_SIZE):
                    egx = eg * GUESS_SIZE
                    correct = [first_guess] + guess_results[egx:egx + GUESS_SIZE]
                    myguess = guess_tokens[egx:egx + GUESS_SIZE]
                    gg = 0
                    for gg in range(len(myguess)):
                        if myguess[gg] != correct[gg]:
                            break 
                    if gg > max_hit:
                        max_hit = gg 
                        max_hit_idx = eg 
                        hits[:max_hit + 1] = correct[:max_hit + 1]
            #max_hit is the length of longest accepted sequence in verification branch 
            
            # lookahead分支的输出
            new_results = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()
            
            assert len(past_tokens[LEVEL - 2]) == WINDOW_SIZE and len(new_results) == WINDOW_SIZE
            # lst_token是当前推理得到的最后一个正确的token，
            # 通过past_tokens，new_results更新token_map中lst_token对应的n-gram
            # 如果设置了GUESS_SET_SIZE则使用LRU机制淘汰最久未使用的
            update_token_map(token_map, lst_token, past_tokens, new_results, LEVEL, WINDOW_SIZE, GUESS_SET_SIZE)

            # 在past_tokens 全部填满之后会用新的覆盖老的，
            # 首先把前N-2个向前移动一位，然后把最新的赋值到最后一位
            if ALWAYS_FWD_ONE:
                past_tokens[0] = past_tokens[1][1:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][:]

                past_tokens[LEVEL - 2] = new_results             
            else:
                past_tokens[0] = past_tokens[1][1 + max_hit:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][max_hit:]

                past_tokens[LEVEL - 2] = new_results[max_hit:]

        if max_hit > 0:
            if not ALWAYS_FWD_ONE:
                for level in range(LEVEL - 1):
                    past_tokens[level] = past_tokens[level] + [set_token() for _ in range(max_hit)]

            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat((attention_mask, torch.ones(1, max_hit, device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)
    

        past_key_values = []

        #plan to remove kv-cache copy and set tokens into next input when dist_workers > 1, as communication is costly
        guess_skip_dist = 0
        offset_kv_cache = outputs.step_len-len(guess_tokens)+max_hit_idx * GUESS_SIZE if max_hit > 0 else 0
        # len(outputs.past_key_values)代表的是模型的层数，TinyLlama/TinyLlama-1.1B-Chat-v1.0有22层
        for idx, kv in enumerate(outputs.past_key_values):
            #update kv-cache from verification branch，通过命中的长度更新kvcavhe
            if max_hit > 0:
                kv[0][:,:,outputs.kvcache_len:outputs.kvcache_len+max_hit,:] = kv[0][:,:,offset_kv_cache:offset_kv_cache+max_hit,:]
                kv[1][:,:,outputs.kvcache_len:outputs.kvcache_len+max_hit,:] = kv[1][:,:,offset_kv_cache:offset_kv_cache+max_hit,:]
            past_key_values.append( (kv[0][:,:,:outputs.kvcache_len + max_hit,:], kv[1][:,:,:outputs.kvcache_len + max_hit,:]) )
        outputs.past_key_values = past_key_values

        # 设置最后一个正确的token
        lst_token = hits[max_hit]

        #stopping condition
        for hit_idx in range(max_hit + 1):
            if eos_token_id is not None and hits[hit_idx] == eos_token_id[0]:
                all_old_tokens.append(hits[hit_idx])
                next_tokens = eos_token_id_tensor
                max_hit = hit_idx
                break
            else:
                all_old_tokens.append(hits[max_hit])
                if POOL_FROM_PROMPT:
                    append_new_generated_pool(all_old_tokens[-LEVEL:], token_map, LEVEL, GUESS_SET_SIZE)


        # 将猜测正确的输出拼接到input_ids之后
        input_ids = torch.cat([input_ids, torch.tensor(hits[:max_hit + 1], device=next_tokens.device, dtype=next_tokens.dtype).unsqueeze(0)], dim=-1)
    

        # if eos_token was found in one sentence, set sentence to finished
        # 找到了EOS，结束推理
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break # 从死循环跳出
    
    for criteria in stopping_criteria:
        if hasattr(criteria, "max_length"):
            #print("steop: ",  criteria.max_length, init_len, len(all_old_tokens), input_ids.size())
            all_old_tokens = all_old_tokens[:criteria.max_length]
            input_ids = input_ids[:,:criteria.max_length]
    if max_length is not None:
        #print("max : ", max_length, init_len)
        all_old_tokens = all_old_tokens[:init_len + max_length]
        input_ids = input_ids[:][:init_len + max_length]
    #返回输出结果
    return input_ids
