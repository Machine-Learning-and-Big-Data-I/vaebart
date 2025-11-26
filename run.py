from transformers import BartTokenizer
from vaebart import VAEBartForConditionalGeneration
from fastai.text.all import *
from blurr.text.data.all import *
from blurr.text.modeling.all import *
import nltk
nltk.download('punkt')
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import ast
import torch
from evaluate import load

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

class SimplificationDataset(torch.utils.data.Dataset):
    def __init__(self, source_sentences, target_sentences, tokenizer):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source_sentence = self.source_sentences[idx]
        target_sentence = self.target_sentences[idx]

        source_encoding = self.tokenizer(source_sentence, max_length=1024, padding="max_length", truncation=True, return_tensors='pt')
        target_encoding = self.tokenizer(target_sentence, max_length=1024, padding="max_length", truncation=True, return_tensors='pt')


        return {
            'input_ids': source_encoding['input_ids'][0],
            'attention_mask': source_encoding['attention_mask'][0],
            'target_ids': target_encoding['input_ids'][0],
            'target_attention_mask': target_encoding['attention_mask'][0]
            }

df_train = pd.read_csv('train_ASSET_preprocess.csv')

# Import model
# Source: https://ohmeow.github.io/blurr/text.data.seq2seq.core.html
pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(pretrained_model_name, model_cls=VAEBartForConditionalGeneration)

hf_config.max_length=70
hf_config.min_length=10
hf_config.no_repeat_ngram_size=2
hf_config.num_beams=5

text_gen_kwargs = {}
if hf_arch in ["bart", "t5"]:
    text_gen_kwargs = {**hf_config.task_specific_params["summarization"], **{"max_length": 50, "min_length": 10}}

# not all "summarization" parameters are for the model.generate method ... remove them here
generate_func_args = list(inspect.signature(hf_model.generate).parameters.keys())
for k in text_gen_kwargs.copy():
    if k not in generate_func_args:
        del text_gen_kwargs[k]

if hf_arch == "mbart":
    text_gen_kwargs["decoder_start_token_id"] = hf_tokenizer.get_vocab()["en_XX"]

tok_kwargs = {}
if hf_arch == "mbart":
    tok_kwargs["src_lang"], tok_kwargs["tgt_lang"] = "en_XX", "en_XX"

batch_tokenize_tfm = Seq2SeqBatchTokenizeTransform(
    hf_arch,
    hf_config,
    hf_tokenizer,
    hf_model,
    max_length=256,
    max_target_length=130,
    tok_kwargs=tok_kwargs,
    text_gen_kwargs=text_gen_kwargs,
)

blocks = (Seq2SeqTextBlock(batch_tokenize_tfm=batch_tokenize_tfm), noop)

dblock = DataBlock(blocks=blocks, get_x=ColReader("original"), get_y=ColReader("simplifications"), splitter=RandomSplitter())
dls = dblock.dataloaders(df_train, bs=2)

seq2seq_metrics = {
    "rouge": {
        "compute_kwargs": {"rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"], "use_stemmer": True},
        "returns": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    },
    "bertscore": {"compute_kwargs": {"lang": "en"}, "returns": ["precision", "recall", "f1"]},
}

# Training
# Source: https://github.com/ohmeow/blurr
model = BaseModelWrapper(hf_model)
learn_cbs = [BaseModelCallback]
fit_cbs = [Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

learn = Learner(
    dls,
    model,
    opt_func=partial(Adam),
    loss_func=CrossEntropyLossFlat(),
    cbs=learn_cbs,
    splitter=partial(blurr_seq2seq_splitter, arch='bart'),
)

learn.freeze()
learn.summary()
learn.lr_find(suggest_funcs=[minimum, steep, valley, slide])
learn.fit_one_cycle(2, lr_max=1e-05, cbs=fit_cbs)

#Eval
df_test = pd.read_csv('test_ASSET_preprocess.csv')

#Use test data to generate simplifications
df_test['simplified_snt'] = " "

#generate simplifications
def simplify (snt):
  output = learn.blurr_generate(snt, num_return_sequences=1)
  final = output[0]['generated_texts']
  return(final)

df_test['simplified_snt'] = df_test.swifter.apply(lambda row: simplify(row['original']),axis=1)
df_test.to_csv('testTSVAEBART_DualMeanPooling.csv')

# Calculating Rouge
df = df_test
for index, row in df.iterrows():
    reference = row['simplifications']
    generated = row['simplified_snt']

    scores = rouge.get_scores(generated, reference)

    # Access the ROUGE scores
    rouge_1 = scores[0]['rouge-1']['f']
    rouge_2 = scores[0]['rouge-2']['f']
    rouge_l = scores[0]['rouge-l']['f']

    # Do something with the scores, such as storing them in a new column of the dataframe
    df.loc[index, 'rouge-1'] = rouge_1
    df.loc[index, 'rouge-2'] = rouge_2
    df.loc[index, 'rouge-l'] = rouge_l

rouge_1 = np.mean(df['rouge-1'])
rouge_2 = np.mean(df['rouge-2'])
rouge_l = np.mean(df['rouge-l'])

#Calculation BLEU
#source : https://www.nltk.org/_modules/nltk/translate/bleu_score.html
reference = df['simplifications']
candidate = df['simplified_snt']

if len(reference) != len(candidate):
    raise ValueError('The number of sentences in both files do not match.')

score = 0.

for i in range(len(reference)):
  ref = ast.literal_eval(reference[i])
  for j in range(len(ref)):
    score += sentence_bleu([ref[j].strip().split()], candidate[i].strip().split())

    # print(i, reference[i], candidate[i], sentence_bleu([reference[i].strip().split()], candidate[i].strip().split()))

score /= len(reference)*8
print("The bleu score is: "+str(score*100))

#Calculating BERTSCORE
bertscore = load("bertscore")
reference = df['simplifications'].to_list()
candidate = df['simplified_snt'].to_list()
# bertscore = BERTScore()
precision = 0.
recall = 0.
f1 = 0.

for i in range(0,len(reference)):
  ref = ast.literal_eval(reference[i])
  for j in range(0,len(ref)):
    print(candidate[i])
    print(ref[j])
    score = bertscore.compute(predictions=[candidate[i]], references=[ref[j]], lang="en")
    print(score)
    precision += score['precision'][0]
    recall += score['recall'][0]
    f1 += score['f1'][0]

precision /= (len(reference)*10)
print("The precision is: "+str(precision*100))

recall /= (len(reference)*10)
print("The recall is: "+str(recall*100))

f1 /= (len(reference)*10)
print("The f1 is: "+str(f1*100))

#Calculating SARI 
sari = load("sari")
sources=df['original']
predictions=df['simplified_snt']
references=df['simplifications'].to_list()

sari_score = 0.

for i in range(0,len(sources)):
    ref = ast.literal_eval(references[i])
    print(predictions[i])
    print(ref)
    score = sari.compute(sources=[sources[i]], predictions=[predictions[i]], references=[ref])
    print(score)
    sari_score += score['sari']

sari_score /= (len(sources))
print("The sari_score is: "+str(sari_score))