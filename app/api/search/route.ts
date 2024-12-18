import { NextResponse } from 'next/server'
import { Client } from '@elastic/elasticsearch'
import { HfInference } from '@huggingface/inference'
import { z } from 'zod'

// Initialize clients
const hf = new HfInference(process.env.HUGGINGFACE_API_TOKEN)
const es = new Client({
    node: process.env.ELASTICSEARCH_URL,
    auth: {
        apiKey: process.env.ELASTICSEARCH_API_KEY!
    }
})

// Configuration
const MODEL_ID = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
const INDEX_NAME = 'multilingual_content'
const SUPPORTED_LANGUAGES = ['en', 'ar']

// Validation schema
const searchRequestSchema = z.object({
    query: z.string().min(1).max(500),
    limit: z.number().optional().default(10),
    titleWeight: z.number().optional().default(1.5),
    contentWeight: z.number().optional().default(1.0),
})

export async function POST(req: Request) {
    try {
        // Parse and validate request
        const body = await req.json()
        const {
            query,
            limit,
            titleWeight,
            contentWeight,
        } = searchRequestSchema.parse(body)

        // Generate query embedding using Hugging Face
        const queryEmbedding = await hf.featureExtraction({
            model: MODEL_ID,
            inputs: query
        })

        // Build the search query to match the Python indexer's structure
        const searchQuery = {
            index: INDEX_NAME,
            size: limit,
            query: {
                bool: {
                    should: [
                        // Vector similarity scoring - matching Python indexer's structure
                        {
                            script_score: {
                                query: { match_all: {} },
                                script: {
                                    source: `
                                        double score = 0.0;
                                        
                                        if (doc.containsKey('title.vector')) {
                                            double titleSim = cosineSimilarity(params.query_vector, 'title.vector');
                                            score += titleSim * params.title_weight;
                                        }
                                        
                                        return Math.max(0.0, score);
                                    `,
                                    params: {
                                        query_vector: queryEmbedding,
                                        title_weight: 10
                                    }
                                }
                            }
                        },
                        {
                            nested: {
                                path: "content.chunks",
                                score_mode: "max",
                                query: {
                                    script_score: {
                                        query: { match_all: {} },
                                        script: {
                                            source: `
                                                double sim = cosineSimilarity(params.query_vector, 'content.chunks.vector');
                                                return Math.max(0, sim * params.content_weight);
                                            `,
                                            params: {
                                                query_vector: queryEmbedding,
                                                content_weight: 7
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            highlight: {
                pre_tags: ['<mark>'],
                post_tags: ['</mark>'],
                fields: {
                    'title.original': {},
                    'content.original': {
                        fragment_size: 150,
                        number_of_fragments: 3
                    },
                    ...Object.fromEntries(
                        SUPPORTED_LANGUAGES.flatMap(lang => [
                            [`title.translations.${lang}`, {}],
                            [`content.translations.${lang}`, {
                                fragment_size: 150,
                                number_of_fragments: 3
                            }]
                        ])
                    )
                }
            },
            _source: ['url', 'title', 'content', 'original_language', 'domain', 'timestamp'],
        }

        const response = await es.search(searchQuery)

        // Format results to match the indexed document structure
        const results = response.hits.hits.map(hit => {
            const source = hit._source as any
            const highlight = hit.highlight || {}

            // Get highlighted content, preferring chunked content matches
            const getHighlightedContent = () => {
                // Try to get highlight from chunks first
                const contentHighlight = highlight['content.original']?.[0] ||
                    source.content.chunks?.[0]?.text?.substring(0, 200)

                if (contentHighlight) {
                    return {
                        text: contentHighlight,
                        language: source.original_language || 'unknown'
                    }
                }

                // Check translations
                for (const lang of SUPPORTED_LANGUAGES) {
                    const translationHighlight = highlight[`content.translations.${lang}`]?.[0]
                    if (translationHighlight) {
                        return {
                            text: translationHighlight,
                            language: lang
                        }
                    }
                }

                return {
                    text: source.content.original?.substring(0, 200) + '...',
                    language: source.original_language || 'unknown'
                }
            }

            const getHighlightedTitle = () => {
                if (highlight['title.original']?.[0]) {
                    return {
                        text: highlight['title.original'][0],
                        language: ''
                    }
                }

                for (const lang of SUPPORTED_LANGUAGES) {
                    const translationHighlight = highlight[`title.translations.${lang}`]?.[0]
                    if (translationHighlight) {
                        return {
                            text: translationHighlight,
                            language: lang
                        }
                    }
                }

                return {
                    text: source.title.original,
                    language: ''
                }
            }

            const titleContent = getHighlightedTitle()
            const mainContent = getHighlightedContent()

            return {
                id: hit._id,
                url: source.url,
                title: {
                    text: titleContent.text,
                    language: titleContent.language
                },
                content: {
                    text: mainContent.text,
                    language: mainContent.language
                },
                original_language: source.original_language,
                available_translations: Object.keys(source.content.translations || {}),
                domain: source.domain,
                timestamp: source.timestamp,
                score: hit._score || 0
            }
        })

        return NextResponse.json({
            results,
            total: response.hits.total,
            query_time_ms: response.took,
            languages: SUPPORTED_LANGUAGES
        })

    } catch (error) {
        console.error('Search error:', error)

        if (error instanceof z.ZodError) {
            return NextResponse.json(
                { error: 'Invalid request parameters', details: error.issues },
                { status: 400 }
            )
        }

        return NextResponse.json(
            { error: 'Internal server error', details: error instanceof Error ? error.message : 'Unknown error' },
            { status: 500 }
        )
    }
}