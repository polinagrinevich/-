# Код и данные для курсовой работы "Измерение установок русскоязычных пользователей социальной сети ВКонтакте в отношении социальных и политических групп на основе текстовых данных: разработка и апробация методики"


## Сбор данных
Код для сбора данных: `vk_parsing.ipynb`

Первичная преобработка: `data_preprocessing.ipynb`

Полный массив собранных данных (>400 000 комментариев): (`comm_with_posts_final_10_3.parquet`) https://disk.yandex.ru/d/HLaXJNNvqYPDjg 

## Фильтрация данных

Код для отбора релевантных комментариев и постов: `data_filtering.csv`

Полученный очищеннный массив данных: `filtered_pol_df.csv` (https://disk.yandex.ru/d/HLaXJNNvqYPDjg)

## Обучение модели

Тренировочный массив данных, тестовый массив данных:  `df_train_pol.csv`, `df_test_pol.csv`

Код для обучения: `affect_combined_bert.ipynb`

Преобученная модель: `model_FINAL` (https://disk.yandex.ru/d/HLaXJNNvqYPDjg)

## Код для анализа
Построение графиков и регрессий: `data_analysis.ipynb`



Код для использования предобученной модели: 

```
class BertForMultiTaskClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)


        self.classifier1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 3),
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 3),
        )

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels1=None, labels2=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]

        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)

        return logits1, logits2

model = BertForMultiTaskClassification.from_pretrained(checkpoint, config=config)
model.load_state_dict(torch.load('model_FINAL'))
model = model.to(device)

checkpoint = "DeepPavlov/rubert-base-cased-conversational"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize(data, tokenizer, loader=True, batch_size=16, shuffle=True, label=True):

    data_tokenized = []
    
    for i, row in tqdm(data.iterrows()):
        inputs = tokenizer.encode_plus(
                row['text'],
                add_special_tokens=True,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_attention_mask = True,
                return_tensors = 'pt',
            )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        if label:
            labels_isr = row['affect_isr']
            labels_pal = row['affect_pal']
            final = {
                    'input_ids': ids.clone().detach(),
                    'attention_mask': mask.clone().detach(),
                    'labels_isr': torch.tensor(labels_isr, dtype=torch.float),
                    'labels_pal': torch.tensor(labels_pal, dtype=torch.float)
                }
            
        else:
            final = {
                    'input_ids': ids.clone().detach(),
                    'attention_mask': mask.clone().detach()
                }

        data_tokenized.append(final)


        dataloader = DataLoader(data_tokenized, batch_size=16, shuffle=shuffle)
    
    if loader:
        return dataloader
    else:
        return data_tokenized


def evaluate(test, model):
    logits_isr, logits_pal = [], []
    preds_isr, preds_pal = [], []

    for i in tqdm(test):
    
        ids = i['input_ids'].clone().detach().to(device)
        mask = i['attention_mask'].clone().detach().to(device)
        
        with torch.no_grad():
            output = model(ids, mask)
            logit_isr, logit_pal = output
            logit_isr, logit_pal = logit_isr.detach().cpu().numpy(), logit_pal.detach().cpu().numpy()
            
            logits_isr.append(logit_isr[0])
            logits_pal.append(logit_pal[0])
            preds_isr.append(np.argmax(logit_isr, axis=1))
            preds_pal.append(np.argmax(logit_pal, axis=1))

    return logits_isr, logits_pal, preds_isr, preds_pal


test = tokenize(df_full, tokenizer, shuffle=False, loader=False)
logits_isr, logits_pal, preds_isr, preds_pal = evaluate(test, model)
```

Массив данных с предсказанными значениями: `full_df_pred` (https://disk.yandex.ru/d/HLaXJNNvqYPDjg)




