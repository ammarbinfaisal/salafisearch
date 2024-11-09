import { Search, Sparkles, ChevronRight } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from "@/components/ui/badge";
import { Client } from '@elastic/elasticsearch';
import { HfInference } from '@huggingface/inference';
import { z } from 'zod';
import Head from 'next/head';

// Initialize clients
const hf = new HfInference(process.env.HUGGINGFACE_API_TOKEN);
const es = new Client({
  node: process.env.ELASTICSEARCH_URL,
  auth: {
    apiKey: process.env.ELASTICSEARCH_API_KEY!
  }
});

// Configuration
const MODEL_ID = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2';
const INDEX_NAME = 'multilingual_content';
const SUPPORTED_LANGUAGES = ['en', 'ar'];

// Validation schema
const searchRequestSchema = z.object({
  query: z.string().min(1).max(500),
  limit: z.number().optional().default(10),
  titleWeight: z.number().optional().default(1.5),
  contentWeight: z.number().optional().default(1.0),
});

const LANGUAGE_NAMES: Record<string, string> = {
  en: "English",
  ar: "Arabic",
  ur: "Urdu",
};

const truncate = (text: string, length: number): string => {
  if (text.length <= length) return text;
  return text.slice(0, length) + '...';
}

async function performSearch(query: string) {
  try {
    const params = searchRequestSchema.parse({
      query,
      limit: 30,
      titleWeight: 1.5,
      contentWeight: 1.2,
    });

    // Generate query embedding using Hugging Face
    const queryEmbedding = await hf.featureExtraction({
      model: MODEL_ID,
      inputs: params.query
    });

    // Build the search query
    const searchQuery = {
      index: INDEX_NAME,
      size: params.limit,
      query: {
        bool: {
          should: [
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
    };

    const response = await es.search(searchQuery);
    return formatSearchResults(response);
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
}

function formatSearchResults(response: any) {
  const results = response.hits.hits.map((hit: any) => {
    const source = hit._source;
    const highlight = hit.highlight || {};

    const getHighlightedContent = () => {
      const contentHighlight = highlight['content.original']?.[0] ||
        source.content.chunks?.[0]?.text?.substring(0, 200);

      if (contentHighlight) {
        return {
          text: contentHighlight,
          language: source.original_language || 'unknown'
        };
      }

      for (const lang of SUPPORTED_LANGUAGES) {
        const translationHighlight = highlight[`content.translations.${lang}`]?.[0];
        if (translationHighlight) {
          return {
            text: translationHighlight,
            language: lang
          };
        }
      }

      return {
        text: source.content.original?.substring(0, 200) + '...',
        language: source.original_language || 'unknown'
      };
    };

    const getHighlightedTitle = () => {
      if (highlight['title.original']?.[0]) {
        return {
          text: highlight['title.original'][0],
          language: ''
        };
      }

      for (const lang of SUPPORTED_LANGUAGES) {
        const translationHighlight = highlight[`title.translations.${lang}`]?.[0];
        if (translationHighlight) {
          return {
            text: translationHighlight,
            language: lang
          };
        }
      }

      return {
        text: source.title.original,
        language: ''
      };
    };

    const titleContent = getHighlightedTitle();
    const mainContent = getHighlightedContent();

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
    };
  });

  return {
    results,
    total: response.hits.total,
    query_time_ms: response.took,
    languages: SUPPORTED_LANGUAGES
  };
}

async function getStats() {
  const response = await es.indices.stats({ index: INDEX_NAME });
  return {
    document_count: response._all.total?.docs?.count,
    index_size_bytes: response._all.total?.store?.size_in_bytes,
    languages: SUPPORTED_LANGUAGES
  };
}

function SearchForm({ defaultQuery = '' }: { defaultQuery?: string }) {
  return (
    <form action="/" method="GET" className="relative">
      <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
        <Search className="h-4 w-4 text-gray-400" />
      </div>
      <Input
        name="q"
        placeholder="Search across languages..."
        defaultValue={defaultQuery}
        className="pl-10 h-12 text-lg shadow-sm"
      />
      <Button
        type="submit"
        className="absolute right-1.5 top-1.5 h-9"
      >
        <div className="flex items-center">
          <Sparkles className="w-4 h-4 mr-2" />
          Search
        </div>
      </Button>
    </form>
  );
}

function SearchResults({ results, stats, queryTime }: any) {
  const formatAsBreadcrumbs = (url: string) => {
    try {
      const urlObj = new URL(url);
      const pathParts = urlObj.pathname.split(/\/|\?/).filter(part => part);

      return (
        <div className="flex items-center text-sm text-gray-600 overflow-x-auto">
          <span className="text-gray-500">{urlObj.hostname}</span>
          {pathParts.length > 0 && <ChevronRight className="w-4 h-4 mx-1 text-gray-400" />}
          {pathParts.map((part: string, index: number) => (
            <div key={index} className="flex items-center">
              <span className="hover:text-gray-800">{truncate(decodeURIComponent(part), 10)}</span>
              {index < pathParts.length - 1 && (
                <ChevronRight className="w-4 h-4 mx-1 text-gray-400" />
              )}
            </div>
          ))}
        </div>
      );
    } catch {
      return <span className="text-sm text-gray-600">{url}</span>;
    }
  };

  return (
    <>
      <div className="flex flex-wrap items-center gap-4 px-1">
        <span className="text-sm text-gray-500 ml-auto">
          Query Time: {queryTime}ms
        </span>
      </div>

      {results.length > 0 ? (
        <div className="space-y-4">
          {results.map((result: any, index: number) => (
            <Card key={index} className="overflow-hidden hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="space-y-3">
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between items-start gap-4">
                      <div className="space-y-1 min-w-0">
                        <a
                          href={result.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 font-medium block"
                          dangerouslySetInnerHTML={{ __html: result.title.text }}
                        />
                        <div className="flex gap-2 items-center flex-wrap">
                          <Badge variant="secondary" className="bg-blue-50 text-blue-700">
                            {LANGUAGE_NAMES[result.title.language] || result.title.language}
                          </Badge>
                        </div>
                      </div>
                      <Badge variant="outline" className="shrink-0">
                        {result.score.toFixed(2)}
                      </Badge>
                    </div>
                    <a
                      href={result.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline"
                    >
                      {formatAsBreadcrumbs(result.url)}
                    </a>
                  </div>
                  <p
                    className="text-gray-600 text-sm line-clamp-3"
                    dangerouslySetInnerHTML={{ __html: result.content.text }}
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 text-gray-500">
          No results found for your search query.
        </div>
      )}

      <div className="flex gap-6 justify-center text-sm text-gray-500 pt-4">
        <span>{stats?.document_count?.toLocaleString()} documents indexed</span>
        <span>•</span>
        <span>{(stats?.index_size_bytes ? (stats?.index_size_bytes / (1024 * 1024)).toFixed(2) : "")} MB index size</span>
      </div>
    </>
  );
}

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ q?: string }>
}) {
  // Pre-fetch data on the server
  const query = (await searchParams).q;
  const [searchData, stats] = query 
    ? await Promise.all([performSearch(query), getStats()])
    : [null, await getStats()];

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white p-6">
      <Head>
        <title>Salafi Search | {query || ''}</title>
        <meta name="description" content="Search engine for Islamic content" />
      </Head>
      <div className="max-w-4xl mx-auto space-y-6">
        <SearchForm defaultQuery={query} />
        
        {query && searchData && (
          <SearchResults 
            results={searchData.results}
            stats={stats}
            queryTime={searchData.query_time_ms}
          />
        )}

        {!query && (
          <div className="text-center py-12 text-gray-500">
            Enter a search query to begin...
          </div>
        )}
      </div>
    </div>
  );
}