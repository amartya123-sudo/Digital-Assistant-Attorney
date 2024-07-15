import torch
from transformers import DPRReader, DPRReaderTokenizer, DPRContextEncoderTokenizer

class DPR:
    def __init__(self, context, question):
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.rdr_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        self.rdr_model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
        self.context = context
        self.question = question
        self.answer = None
        self.source = None
    
    def _preprocessing(self, max_length=256):
        sentences = self.context.split('. ')
        tokenized_sentences = [self.ctx_tokenizer(sentence, return_tensors='pt')['input_ids'] for sentence in sentences]
        sentence_lengths = [len(tokens[0]) for tokens in tokenized_sentences]

        chunks = []
        current_chunk = []
        current_length = 0

        for tokens, length in zip(tokenized_sentences, sentence_lengths):
            if current_length + length > max_length:
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0

            current_chunk.append(tokens)
            current_length += length

        if current_chunk:
            chunks.append(current_chunk)

        chunked_token_ids = [torch.cat(chunk, dim=1) for chunk in chunks]
        return chunked_token_ids

    def _retriever(self):
        self.answer = ""
        score = -float('inf')
        self.source = ""

        chunks = self._preprocessing()
        for chunk in chunks:
            text = self.rdr_tokenizer.decode(chunk[0], skip_special_tokens=True)
            encoded_inputs = self.rdr_tokenizer(
                questions=[self.question],
                texts=[text],
                max_length=512,
                return_tensors="pt",
                truncation=True,
            )
            outputs = self.rdr_model(**encoded_inputs)

            softmax = torch.nn.Softmax(dim=-1)
            start_logits = softmax(outputs.start_logits[0])
            end_logits = softmax(outputs.end_logits[0])

            start_index = torch.argmax(start_logits).item()
            end_index = torch.argmax(end_logits).item()
            an_score = start_logits[start_index] + end_logits[end_index]

            if an_score > score:
                score = an_score
                span = encoded_inputs["input_ids"][0][start_index:end_index + 1]
                self.answer = self.rdr_tokenizer.decode(span, skip_special_tokens=True)
                self.source = text

        return {'answer':self.answer, 'source':self.source}

    def __call__(self):
        return self._retriever()

"""What is the total compensation amount that the appellants are entitled to?"""
