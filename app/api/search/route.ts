import { NextRequest, NextResponse } from 'next/server';
import client from '@/lib/elasticsearch';

export async function POST(req: NextRequest) {
    try {
        const { query, size = 10, min_score = 0.5 } = await req.json();

        const queryHasArabic = /[\u0600-\u06FF]/.test(query);

        const shouldClauses = [];
        const supportedLanguages = (process.env.SUPPORTED_LANGUAGES || 'en,ar').split(',');

        for (const lang of supportedLanguages) {
            const weight = lang === 'ar' && queryHasArabic ? 2 : 1;
            shouldClauses.push(
                { match: { [`content.translations.${lang}`]: { query, boost: weight, minimum_should_match: "75%" } } },
                { match: { [`title.translations.${lang}`]: { query, boost: weight * 1.5 } } }
            );
        }

        const response = await client.search({
            index: 'multilingual_content',
            body: {
                query: { bool: { should: shouldClauses, minimum_should_match: 1 } },
                highlight: {
                    fields: supportedLanguages.reduce((acc, lang) => {
                        // @ts-expect-error I expect error here, bruv
                        acc[`content.translations.${lang}`] = { fragment_size: 150, number_of_fragments: 1 };
                        return acc;
                    }, {}),
                    max_analyzed_offset: 500000
                },
                _source: ['url', 'original_language', 'title', 'content', 'domain', 'timestamp'],
                size,
                min_score
            }
        });

        const hits = response.hits.hits;
        const results = hits.map(hit => {
            const source = hit._source;
            // @ts-expect-error I expect error here, bruv
            const originalLanguage = source.original_language;
            // @ts-expect-error I expect error here, bruv
            const snippet = hit.highlight?.[`content.translations.${originalLanguage}`]?.[0] || source.content.translations[originalLanguage].slice(0, 150) + '...';
            return {
                // @ts-expect-error I expect error here, bruv
                url: source.url,
                score: hit._score,
                // @ts-expect-error I expect error here, bruv
                original_language: source.original_language,
                // @ts-expect-error I expect error here, bruv
                title: source.title.translations[originalLanguage] || '',
                snippet,
                // @ts-expect-error I expect error here, bruv
                translations: source.content.translations,
                // @ts-expect-error I expect error here, bruv
                domain: source.domain,
                // @ts-expect-error I expect error here, bruv
                timestamp: new Date(source.timestamp).toISOString()
            };
        });

        return NextResponse.json({
            results,
            // @ts-expect-error I expect error here, bruv
            total_hits: response.hits.total.value,
            query_time_ms: response.took,
            // @ts-expect-error I expect error here, bruv
            matched_languages: [...new Set(hits.map(hit => hit._source.original_language))]
        });
    } catch (error) {
        console.error('Search error:', error);
        return NextResponse.json({ error: 'Error during search' }, { status: 500 });
    }
}
