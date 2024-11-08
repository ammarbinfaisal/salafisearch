import { NextResponse } from 'next/server';
import client from '@/lib/elasticsearch';

export async function GET() {
    try {
        const esHealth = await client.cluster.health();
        return NextResponse.json({
            status: 'healthy',
            elasticsearch: esHealth.status,
            supported_languages: (process.env.SUPPORTED_LANGUAGES || 'en,ar').split(',')
        });
    } catch (error) {
        console.error('Health check error:', error);
        return NextResponse.json({ error: 'System unhealthy' }, { status: 503 });
    }
}
