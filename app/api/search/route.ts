import { NextRequest, NextResponse } from 'next/server';
import client from '@/lib/elasticsearch';

export async function POST(req: NextRequest) {
    try {
        const { query, size = 10, min_score = 0.5, language_weights } = await req.json();

        const shouldClauses = [];
        const supportedLanguages = (process.env.SUPPORTED_LANGUAGES || 'en,ar').split(',');

        for (const lang of supportedLanguages) {
            const weight = language_weights?.[lang] || 1.0;
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
                        // @ts-ignore
                        acc[`content.translations.${lang}`] = { fragment_size: 150, number_of_fragments: 1 };
                        return acc;
                    }, {}),
                    pre_tags: ['<em>'],
                    post_tags: ['</em>'],
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
            // @ts-ignore
            const originalLanguage = source.original_language;
            // @ts-ignore
            const snippet = hit.highlight?.[`content.translations.${originalLanguage}`]?.[0] || source.content.translations[originalLanguage].slice(0, 150) + '...';
            return {
                // @ts-ignore
                url: source.url,
                score: hit._score,
                // @ts-ignore
                original_language: source.original_language,
                // @ts-ignore
                title: source.title.translations[originalLanguage] || '',
                snippet,
                // @ts-ignore
                translations: source.content.translations,
                // @ts-ignore
                domain: source.domain,
                // @ts-ignore
                timestamp: new Date(source.timestamp).toISOString()
            };
        });

        return NextResponse.json({
            results,
            // @ts-ignore
            total_hits: response.hits.total.value,
            query_time_ms: response.took,
            // @ts-ignore
            matched_languages: [...new Set(hits.map(hit => hit._source.original_language))]
        });
    } catch (error) {
        console.error('Search error:', error);
        return NextResponse.json({ error: 'Error during search' }, { status: 500 });
    }
}
