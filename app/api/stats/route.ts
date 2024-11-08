import { NextResponse } from 'next/server';
import client from '@/lib/elasticsearch';

export async function GET() {
    try {
        const stats = await client.indices.stats({ index: 'multilingual_content' });
        const indexInfo = await client.indices.get({ index: 'multilingual_content' });

        // @ts-ignore
        const documentCount = stats.indices.multilingual_content.total.docs.count;
        
        // @ts-ignore
        const indexSizeBytes = stats.indices.multilingual_content.total.store.size_in_bytes;


        // @ts-ignore
        const creationDate = new Date(parseInt(indexInfo.multilingual_content.settings.index.creation_date)).toISOString();

        return NextResponse.json({
            document_count: documentCount,
            index_size_bytes: indexSizeBytes,
            languages: (process.env.SUPPORTED_LANGUAGES || 'en,ar').split(','),
            creation_date: creationDate
        });
    } catch (error) {
        console.error('Stats error:', error);
        return NextResponse.json({ error: 'Error fetching stats' }, { status: 500 });
    }
}
