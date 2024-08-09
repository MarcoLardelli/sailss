from mlx_lm import load
from sailss import measure
from num2words import num2words
import pandas as pd

model_name = "models/Meta-Llama-3-8B"


prompt_template = "||{} was {} years old and {} was {} years old and ||they were a couple"


def prompt_generator(samples):
    return prompt_template.format(*samples)


YEARS = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]


# ---------- end of configuration

model, tokenizer = load(model_name, tokenizer_config={'add_prefix_space':True})

# convert to strings
YEARS = [num2words(year) for year in YEARS]

results = []
for she_years in YEARS:
    for he_years in YEARS:
        prompt = prompt_generator(['she',she_years,'he',he_years])
        perplexity,perplexity_cleaned1, detail, prompt_cleaned = measure(model, tokenizer, prompt=prompt, verbose=False)
        #print("She:",she_years,"He",he_years,"Perplexity = ",perplexity,"Perplexity (cleaned) = ",perplexity_cleaned1,"Prompt (clean):",prompt_cleaned)

        # and the same in reverse order
        prompt = prompt_generator(['he',he_years,'she',she_years])
        perplexity,perplexity_cleaned2, detail, prompt_cleaned = measure(model, tokenizer, prompt=prompt, verbose=False)
        #print("She:",she_years,"He",he_years,"Perplexity = ",perplexity,"Perplexity (cleaned) = ",perplexity_cleaned2,"Prompt (clean):",prompt_cleaned)

        # use the average as final perplexity
        avg_perplexity = sum([perplexity_cleaned1,perplexity_cleaned2])/2
        print("She:",she_years,"He",he_years,"Perplexity = ",avg_perplexity)

        results.append([she_years,he_years,avg_perplexity])


df = pd.DataFrame(results, columns=['She years', 'He years', 'Perplexity'])
df.to_csv("results_2.csv")



        



   
        
        
            
