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
    matchPhrase: z.boolean().optional().default(false),
    language: z.string().optional(),
})

const isArabic = (text: string) => {
    return /[\u0600-\u06FF]/.test(text)
}

export async function POST(req: Request) {
    try {
        // Parse and validate request
        const body = await req.json()
        const {
            query,
            limit,
            titleWeight,
            contentWeight,
            matchPhrase = false,
            language
        } = searchRequestSchema.parse(body)

        // Generate query embedding using Hugging Face
        const queryEmbedding = await hf.featureExtraction({
            model: MODEL_ID,
            inputs: query
        })

        // Build the search query
        const searchQuery = {
            index: INDEX_NAME,
            size: limit,
            query: {
                bool: {
                    should: [
                        // Text matches on title
                        {
                            bool: {
                                should: [
                                    // Original title match
                                    matchPhrase ? {
                                        match_phrase: {
                                            "title.original": {
                                                query,
                                                boost: titleWeight * 2
                                            }
                                        }
                                    } : {
                                        match: {
                                            "title.original": {
                                                query,
                                                boost: titleWeight * 2,
                                                fuzziness: "AUTO"
                                            }
                                        }
                                    },
                                    // Title translations matches
                                    ...SUPPORTED_LANGUAGES.map(lang => ({
                                        match: {
                                            [`title.translations.${lang}`]: {
                                                query,
                                                boost: titleWeight,
                                                fuzziness: "AUTO"
                                            }
                                        }
                                    }))
                                ],
                            }
                        },
                        // Text matches on content
                        {
                            bool: {
                                should: [
                                    // Original content match
                                    matchPhrase ? {
                                        match_phrase: {
                                            "content.original": {
                                                query,
                                                boost: contentWeight
                                            }
                                        }
                                    } : {
                                        match: {
                                            "content.original": {
                                                query,
                                                boost: contentWeight,
                                                fuzziness: "AUTO"
                                            }
                                        }
                                    },
                                    // Content translations matches
                                    ...SUPPORTED_LANGUAGES.map(lang => ({
                                        match: {
                                            [`content.translations.${lang}`]: {
                                                query,
                                                boost: contentWeight,
                                                fuzziness: "AUTO"
                                            }
                                        }
                                    }))
                                ]
                            }
                        },
                        // Vector similarity on title
                        {
                            script_score: {
                                query: { match_all: {} },
                                script: {
                                    source: "cosineSimilarity(params.query_vector, 'title.vector') + 1.0",
                                    params: { query_vector: queryEmbedding }
                                }
                            }
                        }
                    ],
                    minimum_should_match: 1
                }
            },
            highlight: {
                pre_tags: ['<mark>'],
                post_tags: ['</mark>'],
                fields: {
                    'title.original': {},
                    'content.original': {
                        fragment_size: 150,
                        number_of_fragments: 1
                    },
                    ...Object.fromEntries(
                        SUPPORTED_LANGUAGES.flatMap(lang => [
                            [`title.translations.${lang}`, {}],
                            [`content.translations.${lang}`, {
                                fragment_size: 150,
                                number_of_fragments: 1
                            }]
                        ])
                    )
                }
            }
        }

        const response = await es.search(searchQuery)

        // Format results
        const results = response.hits.hits.map(hit => {
            const source = hit._source as any
            const highlight = hit.highlight || {}

            // Get the best matching content version
            const getHighlightedContent = () => {
                // Check original content first
                if (highlight['content.original']?.[0]) {
                    return {
                        text: highlight['content.original'][0],
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

                // Fallback to original content
                return {
                    text: source.content.original?.substring(0, 200) + '...',
                    language: isArabic(source.content.original) ? 'ar' : 'en'
                }
            }

            // Get the best matching title version
            const getHighlightedTitle = () => {
                if (highlight['title.original']?.[0]) {
                    return {
                        text: highlight['title.original'][0],
                        language: source.original_language || 'unknown'
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
                    language: isArabic(source.content.original) ? 'ar' : 'en'
                }
            }

            const titleContent = getHighlightedTitle()
            const mainContent = getHighlightedContent()

            return {
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