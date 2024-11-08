// app/api/embed-local/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { join } from 'path';
import * as ort from 'onnxruntime-web';
import { tokenize } from './tokenizer';  // We'll create this

// Initialize ONNX Session (load only once)
let session: ort.InferenceSession | null = null;

async function initializeSession() {
    if (!session) {
        // Model should be in the public/models directory
        const modelPath = join(process.cwd(), 'public', 'models', 'model.onnx');
        session = await ort.InferenceSession.create(modelPath);
    }
    return session;
}

export async function POST(req: NextRequest) {
    try {
        const { text } = await req.json();

        if (!text) {
            return NextResponse.json(
                { error: 'Text is required' },
                { status: 400 }
            );
        }

        // Initialize session if needed
        const sess = await initializeSession();

        // Tokenize input (using transformer tokenizer)
        const tokenized = await tokenize(text);

        // Create ONNX tensor
        const inputTensor = new ort.Tensor(
            'int64',
            tokenized.input_ids,
            [1, tokenized.input_ids.length]
        );

        // Run inference
        const output = await sess.run({
            input_ids: inputTensor
        });

        // Get embeddings from output
        // @ts-expect-error - data is private
        const embedding = Array.from(output.last_hidden_state.data);

        return NextResponse.json({ embedding });
    } catch (error) {
        console.error('Embedding error:', error);
        return NextResponse.json(
            { error: 'Failed to generate embedding' },
            { status: 500 }
        );
    }
}

export const runtime = 'edge';