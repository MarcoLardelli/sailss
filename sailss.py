

"""

#SAILSS: The "Scouting AI Library for Social Sciences"

##Evaluate the likelihood of sentences (or parts of sentences) based on a Large Language Model (LLM)

###Author: Marco Lardelli, Zurich/Switzerland

###Licence: MIT (see included license document)

###Copyright © 2024 Marco Lardelli

This library is described in a series of posts on [my blog](https://lardel.li/2024/07/llm_language_model_library_social_sciences.html)

Github repository: [/MarcoLardelli/sailss](https://github.com/MarcoLardelli/sailss)

"""

import mlx.core as mx
import mlx.nn as nn
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

from mlx_lm.models.base import KVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer

from transformers import PreTrainedTokenizer

import math # added ml






class Experiment:

    """A class to evaluate a number of prompts using the LLM"""

    def __init__(self, model, tokenizer) -> None:
        """
        Create a new experiment, based on a model and a tokenizer
        """
        self.model = model
        """The LLM"""
        self.tokenizer = tokenizer
        """The tokenizer """

        self.prompts = []
        """List of prompts to evaluate"""
        self.results = []
        """List of results"""


    def add_prompt(self, prompt) -> None:
        """
        Add a prompt to the experiment
        """
        self.prompts.append(prompt)


    def _prompts_same_length(self) -> bool:
        """
        Check whether all the prompts have the same length (in tokens!). Returns True if yes, otherwise False
        """
        len0 = len(self.results[0]['details'])
        same_length = True
        for result in self.results:
            details = result['details']
            if len(details)!=len0:
                same_length = False

        return same_length


    def evaluate(self, verbose=False):
        """
        Evaluate all the prompts of this experiment using the LLM model

        Returns:

        - List of results

        Each result is a dict with the following keys:

        - 'prompt': the input prompt
        - 'prompt_cleaned': the input prompt with switch tag || removed and other modifications
        - 'perplexity_full': perplexity of full prompt
        - 'perplexity': perplexity limited to tokes allowed by switch tag ||
        - 'details': list of dicts (see below)


        details is a list of dicts (one for each prompt token) with the following keys:

        - 'index': index of token
        - 'token': the token (int)
        - 'token_str': the token as a string
        - 'logprob': logprob of token
        - 'p': probability of token
        - 'predicted_token_str': token (str) predicted by LLM for this positions


        """
        for prompt in self.prompts:
            perplexity_full,perplexity, details, prompt_cleaned = measure(self.model, self.tokenizer, prompt=prompt, verbose=verbose)
            self.results.append({
                'prompt': prompt,
                'prompt_cleaned': prompt_cleaned,
                'perplexity_full': perplexity_full,
                'perplexity': perplexity,
                'details': details
            })

        return self.results
    
    def _colorprint(color_name, s) -> str:
        """
        Color the string s for console output

        Args:

        - color_name: a string describing the color, like 'red'
        - s: the string to color

        Returns:

        - The colored string
        """
        colorcodes = {
            "black": 30,
            "red": 31,
            "brightred":91,
            "green": 32,
            "brightgreen":92,
            "yellow": 33,
            "brightyellow":93,
            "blue": 34,
            "magenta": 35,
            "cyan": 36,
            "white": 39,
            "gray":252,
        }
        ccode = colorcodes.get(color_name, 30)
        return f"\033[1m\033[{ccode}m{s}\033[0m"


    def _colorprint_by_p(s, p):
        """
        return a string colored (for console) according to probability p

        0 -> 1 corresponds to red (very low probab.) -> yellow (low probab.) -> green (medium probab.) -> blue (high probab.)
        """
        if p==0.5: # a special value returned by normalize()
            color = "gray"
        elif p > 0.75:
            color = "blue"
        elif p > 0.50:
            color = "green"
        elif p > 0.25:
            color = "yellow"
        else:
            color = "red"
        return Experiment._colorprint(color, s)
    

    def _colorprint_by_p_html(s: str, p: float) -> str:
        """
        Wrap string s into a HTML P-tag with a CSS style to add color corresponding to p

        Args:

        - s: a string
        - p: a probability [0,1]

        Returns:

        - A string containing s wrapped in a CSS colored HTML p-Tag 
        """
        if p==0.5:
            color_r = 127
            color_g = 127
            color_b = 127
        else:
            color_r = int((1-p)*255)
            color_g = 0
            color_b = int(p*255)

        hex_color = "#%02x%02x%02x" % (color_r, color_g, color_b)

        return '<p style="color:'+hex_color+'">'+s+'</p>'
    

    def _normalize(p1, plist) -> float:
        """
        Normalize a likelyhood based on a list of likelyhoods

        Args:

        - p1: the likelyhood to normalize
        - plist: a list of likelihoods

        Returns:

        - A normalized likelyhood
        """
        pmin = min(plist)
        pmax = max(plist)
        if (pmax-pmin)>0:
            return (p1-pmin)/(pmax-pmin)
        else:
            return 0.5  # all values of plist are the same -> return this special value (will be rendered gray)
        

    def visualize_results(self):
        """
        print prompt tokens to console (colored by normalized p)
        """
        for d_no,result in enumerate(self.results):
            details = result['details']
         
            out_str=""
            for i,detail in enumerate(details):
                ps = []
                for j,r in enumerate(self.results):
                    ps.append(r['details'][i]['p'])

                normalized_p = Experiment._normalize(detail['p'], ps)

                out_str += Experiment._colorprint_by_p(detail['token_str'][1:], normalized_p)
                out_str +=" "

            print(out_str)
            print()
        

    def create_html_table(self) -> str:
        """
        Create a HTML table containing the results

        Returns:

        - A string containing the HTML table
        """
        if len(self.results)==0 or (not self._prompts_same_length()):
            return None
        
        len0 = len(self.results[0]['details'])

        html = '<table>\n'
        
        for d_no,result in enumerate(self.results):
            tokens = []
            details = result['details']

            probabs = []
            for detail in details:
                probabs.append(detail['p'])

            for token_no in range(len0):
                normlized_probab = Experiment._normalize(details[token_no]['p'],probabs)
                probab = details[token_no]['p']
                token = details[token_no]['token']
                token_str = details[token_no]['token_str']
                predicted_token_str = details[token_no]['predicted_token_str']

                tokens.append(
                    (token_str,normlized_probab,predicted_token_str,token,probab)
                )

            html += '<tr><td>Tokens (str)</td>'
            for i,token in enumerate(tokens):  # tokens, colored by normalized probability
                html += '<td>'+Experiment._colorprint_by_p_html(token[0][1:],token[1])+'</td>'
            html += '</tr>\n'

            html += '<tr><td>Probabs</td>'
            for i,token in enumerate(tokens):  # probabilities
                html += '<td>'+"{:10.8f}".format(token[4])+'</td>'
            html += '</tr>\n'

            html += '<tr><td>Predicted Token (str)</td>'
            for i,token in enumerate(tokens): # predicted token for this position
                if i==0:
                    start=0
                else:
                    start=1
                html += '<td>'+token[2][start:]+'</td>'
            html += '</tr>\n'

            html += '<tr><td>Token (int)</td>'
            for i,token in enumerate(tokens):  # tokens
                html += '<td>'+str(token[3])+'</td>'
            html += '</tr>\n\n'

        html += '</table>\n'

        return html
    

    def save_html_table_to_file(self,file_name:str) -> None:
        """Create a HTML tabel from the results and save it to file_name

        Args:

        - file_name: Name of the file to save the table to
        
        
        """
        html = self.create_html_table()
        with open(file_name, "w") as f:
            print(html, file=f)




def _calculate(
    prompt: mx.array, model: nn.Module,
    tokenizer, token_mask): 

    """

    Args:

    - prompt (mx.array): The input prompt
    - model (nn.Module): The model to use for generation
    - tokenizer: a tokenizer
    - token_mask: a list of bool values indicating which tokens should be included into the perplexity calculation (True = include)

    Returns:

    (perplexity_of_full_sentence,   
    perplexity_of_sentence_parts,   
    details)   

    details is a list of dicts (one for each prompt token) with the following keys:

    - 'index': index of token
    - 'token': the token (int)
    - 'token_str': the token as a string
    - 'logprob': logprob of token
    - 'p': probability of token
    - 'predicted_token_str': token (str) predicted by LLM for this positions
       
    """

    y = prompt
    kv_heads = (
        [model.n_kv_heads] * len(model.layers)
        if isinstance(model.n_kv_heads, int)
        else model.n_kv_heads
    )
    cache = [KVCache(model.head_dim, n) for n in kv_heads]

    def _argmax_chose(logits: mx.array):
        logprobs = logits - mx.logsumexp(logits)

        token = mx.argmax(logits, axis=-1)

        return token, logprobs

    
    def _model_forward_pass(y):
        # we need the logits for ALL prompt words!
        full_logits = model(y[None], cache=cache)
        logits = full_logits[:, -1, :]

        y, logprobs = _argmax_chose(logits)

        return y, logprobs.squeeze(0), full_logits  # we need the full logits too

    y, logprobs, full_logits = _model_forward_pass(y)

    detokenizer = tokenizer.detokenizer

    full_logits2 = full_logits[0,:,:] # we dont need the batch dimension (1)
   
    logprobs_prompt = []
    tokens_prompt = []
    logprobs_sum = 0
    tokens_predicted = []
    
    for i,token in enumerate(prompt):
        if i>0: # all but the first
            logits = full_logits2[i-1]  # the logits from the previous token = prediction for this token
            logprobs = logits - mx.logsumexp(logits)

            predicted_token = mx.argmax(logits, axis=-1)
            tokens_predicted.append(predicted_token.item())
            
            logprob_token = logprobs[token].item()  # here we take now the prompt token and NOT predicted_token !

            tokens_prompt.append(token.item())
            logprobs_prompt.append(logprob_token)

            logprobs_sum += logprob_token

            detokenizer.add_token(token.item())

    logprobs_sum_cleaned = 0
    no_of_clean_tokens = 0
    token_mask_red = token_mask[1:]  # all but the first above
    for i,logprop in enumerate(logprobs_prompt):
        if token_mask_red[i]:
            logprobs_sum_cleaned += logprop
            no_of_clean_tokens += 1

    perplexity = math.exp(- logprobs_sum / len(logprobs_prompt))
    #print("Perplexity:",perplexity)
    perplexity_cleaned = math.exp(- logprobs_sum_cleaned / no_of_clean_tokens)


    prompt_tokens_str = tokenizer.convert_ids_to_tokens(tokens_prompt)
    predicted_tokens_str = tokenizer.convert_ids_to_tokens(tokens_predicted)
   
    details = []
    for i,token_str in enumerate(prompt_tokens_str):
        token = tokens_prompt[i]
        details.append(
            {
                'index':i,
                'token':token,
                'token_str':token_str,
                'logprob': logprobs_prompt[i],
                'p':math.exp(logprobs_prompt[i]),
                'predicted_token_str': predicted_tokens_str[i]
            }

        )

    return perplexity,perplexity_cleaned,details
    



def measure(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    verbose: bool = False,
    **kwargs,
):
    """

    Evaluate a prompt using a Large Language Model

    Args:

    - model (nn.Module): A LLM
    - tokenizer (PreTrainedTokenizer): The tokenizer for this LLM
    - prompt (str): A prompt
    - verbose (bool): if set to True: print various debugging information
    - kwargs: options (getting passed to _calculate function)

    Returns:

    (perplexity_of_full_sentence,   
    perplexity_of_sentence_parts,   
    details,   
    cleaned_prompt)   

    details is a list of dicts (one for each prompt token) with the following keys:

    - 'index': index of token
    - 'token': the token (int)
    - 'token_str': the token as a string
    - 'logprob': logprob of token
    - 'p': probability of token
    - 'predicted_token_str': token (str) predicted by LLM for this positions

    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if verbose:
        print("=" * 10)
        print("Raw prompt:", prompt)

    prompt=prompt.strip() # remove leading and trailing whitespace
    # apply standard tokenizer to prompt  (we need it just for checking)
    cleaned_prompt = prompt.replace('||',' ') # remove the switch tag
    cleaned_prompt = cleaned_prompt.strip() # in case there was a || at the beginning
    cleaned_prompt = cleaned_prompt.replace('  ',' ') # remove double spaces
    cleaned_prompt = ' '+cleaned_prompt # add a single leading space to tokenize the first word in the same way as inside the sentence
    if verbose:
        print("Cleaned prompt:", cleaned_prompt)
    prompt_tokens_list = tokenizer.encode(cleaned_prompt)
    if verbose:
        print("Tokens list (check: =final?):", prompt_tokens_list)
        print("Cleaned prompt,reconstructed:", tokenizer.convert_ids_to_tokens(prompt_tokens_list))

    # now split the raw prompt into segments (divided by the switch || tag) and tokenize segment wise, then detokenize back again to check
    prompt_segments = prompt.split('||')
    prompt_tokens_list_segments = []
    prompt_segments_reconstructed = [] # check if the whole thing works as it should
    for prompt_segment in prompt_segments:
        if prompt_segment=='': # prompt started with a switch tag
            prompt_tokens_list_segments.append(None)
            prompt_segments_reconstructed.append('')
        else:
            p_s = ' '+prompt_segment.strip().replace('  ',' ')  # same defined structure as above
            segment_tokens = tokenizer.encode(p_s)
            prompt_tokens_list_segments.append(segment_tokens)
            prompt_segments_reconstructed.append(tokenizer.convert_ids_to_tokens(segment_tokens))

    if verbose:
        print("Prompt segments token lists:",prompt_tokens_list_segments)
        print("Prompt segments reconstructed:",prompt_segments_reconstructed)

    # now build the tokens list again from the tokenized segments and create the token mask
    token_mask = []
    prompt_tokens_list = []
    prompts_words_list = []
    state = True
    first_valid_segment = True
    for segment_no,segment in enumerate(prompt_tokens_list_segments):
        if segment==None:
            state = False
        else:
            if first_valid_segment:
                segment_corr = segment # inlcude begin_of_text token
                prompt_segment_reconstructed = prompt_segments_reconstructed[segment_no]
                first_valid_segment = False
            else:
                segment_corr = segment[1:]  # remove unwanted begin_of_text token
                prompt_segment_reconstructed = prompt_segments_reconstructed[segment_no][1:]

            prompt_tokens_list += segment_corr
            
            for i,token in enumerate(segment_corr):
                token_mask.append(state)
                prompts_words_list.append(prompt_segment_reconstructed[i])
            state = not state # switch state between segments
        
    if verbose:
        print("Token mask:", token_mask)
        print("Tokens list (final):", prompt_tokens_list)

    # now output all the words data
    
    
    prompt_tokens = mx.array(prompt_tokens_list)

    perplexity,perplexity_cleaned,details = _calculate(prompt_tokens, model, tokenizer, token_mask, **kwargs)

    for i,token in enumerate(prompt_tokens_list):
        if i>0: # ignore the begin_of_text token
            out = details[i-1]
            p = out['p']
            predicted_token_str = out['predicted_token_str']
            if verbose:
                print('#'+str(i),'Token:',token,'=',prompts_words_list[i],"Mask:",token_mask[i],'p=',p,"Predicted:",predicted_token_str)

    return perplexity, perplexity_cleaned, details, cleaned_prompt

    




















