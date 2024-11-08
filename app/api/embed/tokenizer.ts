// app/api/embed-local/tokenizer.ts
import { AutoTokenizer } from '@xenova/transformers';

let tokenizer: any = null;

export async function tokenize(text: string) {
    if (!tokenizer) {
        tokenizer = await AutoTokenizer.from_pretrained(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        );
    }

    return await tokenizer(text, {
        padding: true,
        truncation: true,
        max_length: 512
    });
}