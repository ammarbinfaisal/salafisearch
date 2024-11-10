import { Search, Sparkles, ChevronRight, Languages } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { HfInference } from '@huggingface/inference';
import { Client } from '@elastic/elasticsearch';
import { Metadata } from 'next';
import Head from 'next/head';

// Type definitions
type SupportedLanguage = 'en' | 'ar' | 'ur';

// 1. Add pagination parameters to SearchParams interface
interface SearchParams {
  q?: string;
  lang?: string;
  page?: string;
}
interface ContentWithTranslations {
  original: string;
  translations?: Record<SupportedLanguage, string>;
}

interface DocumentSource {
  url: string;
  title: ContentWithTranslations;
  content: ContentWithTranslations;
  original_language: SupportedLanguage;
  domain: string;
  timestamp: string;
  content_vector: number[];
  content_vector_en?: number[];
  content_vector_ar?: number[];
  content_vector_ur?: number[];
}

interface SearchResult {
  id: string;
  url: string;
  title: {
    text: string;
    language: string;
  };
  content: {
    text: string;
    language: string;
  };
  original_language: SupportedLanguage;
  available_translations: string[];
  domain: string;
  timestamp: string;
  score: number;
}

interface SearchResponse {
  results: SearchResult[];
  total: number;
  query_time_ms: number;
  languages: SupportedLanguage[];
  currentPage: number;
  totalPages: number;
  pageSize: number;
}

interface Stats {
  document_count: number;
  index_size_bytes: number;
  languages: SupportedLanguage[];
}

// Configuration
const MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2';
const INDEX_NAME = 'enhanced';
const SUPPORTED_LANGUAGES: SupportedLanguage[] = ['en', 'ar', 'ur'];
const LANGUAGE_NAMES: Record<SupportedLanguage, string> = {
  en: "English",
  ar: "Arabic",
  ur: "Urdu",
};

// Initialize clients
const hf = new HfInference(process.env.HUGGINGFACE_API_TOKEN!);
const es = new Client({
  node: process.env.ELASTICSEARCH_URL,
  auth: {
    apiKey: process.env.ELASTICSEARCH_API_KEY!
  }
});

// Metadata for the page
export const metadata: Metadata = {
  title: 'Multilingual Search',
  description: 'Enhanced multilingual search engine',
};

const truncate = (text: string, length: number): string => {
  if (text.length <= length) return text;
  return text.slice(0, length) + '...';
};

function SearchForm({ query, language }: { query?: string; language?: string }) {
  return (
    <form action="/enhanced" className="space-y-4">
      <div className="relative">
        <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
          <Search className="h-4 w-4 text-gray-400" />
        </div>
        <Input
          name="q"
          defaultValue={query}
          placeholder="Search across languages..."
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
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Languages className="h-4 w-4 text-gray-500" />
          <Select name="lang" defaultValue={language ?? 'all'}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select Language" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Languages</SelectItem>
              {Object.entries(LANGUAGE_NAMES).map(([code, name]) => (
                <SelectItem key={code} value={code}>{name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </form>
  );
}

function SearchResults({
  results,
  stats,
  queryTime
}: {
  results: SearchResult[];
  stats: Stats;
  queryTime: number;
}) {
  const formatAsBreadcrumbs = (url: string) => {
    try {
      const urlObj = new URL(url);
      const pathParts = urlObj.pathname.split(/\/|\?/).filter(part => part);

      return (
        <div className="flex items-center text-sm text-gray-600 overflow-x-auto">
          <span className="text-gray-500">{urlObj.hostname}</span>
          {pathParts.length > 0 && <ChevronRight className="w-4 h-4 mx-1 text-gray-400" />}
          {pathParts.map((part, index) => (
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
        <div className="flex gap-2 items-center">
          <Badge variant="secondary" className="bg-blue-50 text-blue-700">
            {results.length} results
          </Badge>
          <span className="text-sm text-gray-500">
            in {queryTime}ms
          </span>
        </div>
      </div>

      {results.length > 0 ? (
        <div className="space-y-4">
          {results.map((result, index) => (
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
        <span>{(stats?.index_size_bytes / (1024 * 1024)).toFixed(2)} MB index size</span>
      </div>
    </>
  );
}

type Language = 'en' | 'ar' | 'all' | 'ur';

interface SearchParamsFn {
  queryEmbedding: number[];
  language: Language;
  page: number;
  pageSize: number;
  INDEX_NAME: string;
  query: string;
}

interface HighlightFields {
  [key: string]: {
    fragment_size?: number;
    number_of_fragments?: number;
  };
}

const createSearchQuery = ({
  query,
  queryEmbedding,
  language,
  page,
  pageSize,
  INDEX_NAME
}: SearchParamsFn) => {
  const supportedLanguages = ['en', 'ar'] as const;
  
  // Generate highlight fields based on supported languages
  const highlightFields: HighlightFields = {
    'title.original': {},
    'content.original': {
      fragment_size: 150,
      number_of_fragments: 3
    }
  };

  // Add language-specific highlight fields
  supportedLanguages.forEach(lang => {
    highlightFields[`title.translations.${lang}`] = {};
    highlightFields[`content.translations.${lang}`] = {
      fragment_size: 150,
      number_of_fragments: 3
    };
  });

  return {
    index: INDEX_NAME,
    size: pageSize,
    from: (page - 1) * pageSize,
    query: {
      bool: {
        should: [
          {
            multi_match: {
              query,
              fields: ['title.original^2'],
              minimum_should_match: '95%'
            }
          },
          {
            multi_match: {
              query,
              fields: ['content.original'],
              minimum_should_match: '90%'
            }
          },
          // Vector search on title
          {
            script_score: {
              query: { match_all: {} },
              script: {
                source: "Math.max(0.0, cosineSimilarity(params.query_vector, 'title.vector') * 10 + 1.0)",
                params: { query_vector: queryEmbedding }
              }
            }
          },
          // Vector search on content chunks
          {
            nested: {
              path: "content.chunks",
              score_mode: "max",
              query: {
                script_score: {
                  query: { match_all: {} },
                  script: {
                    source: "Math.max(0.0, cosineSimilarity(params.query_vector, 'content.chunks.vector') + 1.0)",
                    params: { query_vector: queryEmbedding }
                  }
                }
              }
            }
          }
        ],
        // Language-specific boosts
        ...(language !== 'all' && {
          boost: 1.2,
          filter: [
            {
              bool: {
                should: [
                  { term: { original_language: language } },
                  { exists: { field: `content.translations.${language}` } }
                ]
              }
            }
          ]
        })
      }
    },
    highlight: {
      pre_tags: ['<mark>'],
      post_tags: ['</mark>'],
      fields: highlightFields
    },
    _source: [
      'url',
      'domain',
      'id',
      'original_language',
      'timestamp',
      'title.original',
      'title.translations',
      'content.original',
    ]
  };
};


async function performSearch(query: string, page = 1, language: Language = 'all'): Promise<SearchResponse> {
  const pageSize = 10;
  try {
    const queryEmbedding: any = await hf.featureExtraction({
      model: MODEL_ID,
      inputs: query
    });

    const searchQuery  = createSearchQuery({
      query,
      queryEmbedding,
      language,
      page,
      pageSize,
      INDEX_NAME
    });

    const response = await es.search(searchQuery);
    return formatSearchResults(response, language, page, pageSize);
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
}

function formatSearchResults(response: any, preferredLanguage: string, currentPage: number, pageSize: number): SearchResponse {
  const results = response.hits.hits.map((hit: any) => {
    const source = hit._source;
    const highlight = hit.highlight || {};

    const getHighlightedContent = () => {
      if (preferredLanguage !== 'all') {
        const translationHighlight = highlight[`content.translations.${preferredLanguage}`]?.[0];
        if (translationHighlight) {
          return {
            text: translationHighlight,
            language: preferredLanguage
          };
        }
      }

      const contentHighlight = highlight['content.original']?.[0] ||
        source.content.original?.substring(0, 200);

      if (contentHighlight) {
        return {
          text: contentHighlight,
          language: source.original_language
        };
      }

      for (const lang of SUPPORTED_LANGUAGES) {
        if (lang !== preferredLanguage) {
          const translationHighlight = highlight[`content.translations.${lang}`]?.[0];
          if (translationHighlight) {
            return {
              text: translationHighlight,
              language: lang
            };
          }
        }
      }

      return {
        text: source.content.original?.substring(0, 200) + '...',
        language: source.original_language
      };
    };

    const getHighlightedTitle = () => {
      if (preferredLanguage !== 'all' && source.title.translations?.[preferredLanguage]) {
        const translationHighlight = highlight[`title.translations.${preferredLanguage}`]?.[0] ||
          source.title.translations[preferredLanguage];
        return {
          text: translationHighlight,
          language: preferredLanguage
        };
      }

      return {
        text: highlight['title.original']?.[0] || source.title.original,
        language: source.original_language
      };
    };

    const titleContent = getHighlightedTitle();
    const mainContent = getHighlightedContent();

    return {
      id: hit._id,
      url: source.url,
      title: titleContent,
      content: mainContent,
      original_language: source.original_language,
      available_translations: Object.keys(source.content.translations || {}),
      domain: source.domain,
      timestamp: source.timestamp,
      score: hit._score
    };
  });

  return {
    results,
    total: response.hits.total.value,
    query_time_ms: response.took,
    languages: SUPPORTED_LANGUAGES,
    currentPage,
    totalPages: Math.ceil(response.hits.total.value / pageSize),
    pageSize
  };
}

async function getStats(): Promise<Stats> {
  const response = await es.indices.stats({ index: INDEX_NAME });
  return {
    document_count: response._all.total?.docs?.count || 0,
    index_size_bytes: response._all.total?.store?.size_in_bytes || 0,
    languages: SUPPORTED_LANGUAGES
  };
}
function Pagination({
  currentPage,
  totalPages,
  baseUrl
}: {
  currentPage: number;
  totalPages: number;
  baseUrl: string;
}) {
  const pages = [];
  const maxVisible = 5;

  let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
  let endPage = Math.min(totalPages, startPage + maxVisible - 1);

  if (endPage - startPage + 1 < maxVisible) {
    startPage = Math.max(1, endPage - maxVisible + 1);
  }

  return (
    <div className="flex justify-center gap-2 mt-6">
      {currentPage > 1 && (
        <Button
          variant="outline"
          size="sm"
          asChild
        >
          <a href={`${baseUrl}&page=${currentPage - 1}`}>Previous</a>
        </Button>
      )}

      {startPage > 1 && (
        <>
          <Button variant="outline" size="sm" asChild>
            <a href={`${baseUrl}&page=1`}>1</a>
          </Button>
          {startPage > 2 && <span className="px-2">...</span>}
        </>
      )}

      {Array.from({ length: endPage - startPage + 1 }, (_, i) => startPage + i).map(page => (
        <Button
          key={page}
          variant={currentPage === page ? "default" : "outline"}
          size="sm"
          asChild
        >
          <a href={`${baseUrl}&page=${page}`}>{page}</a>
        </Button>
      ))}

      {endPage < totalPages && (
        <>
          {endPage < totalPages - 1 && <span className="px-2">...</span>}
          <Button variant="outline" size="sm" asChild>
            <a href={`${baseUrl}&page=${totalPages}`}>{totalPages}</a>
          </Button>
        </>
      )}

      {currentPage < totalPages && (
        <Button
          variant="outline"
          size="sm"
          asChild
        >
          <a href={`${baseUrl}&page=${currentPage + 1}`}>Next</a>
        </Button>
      )}
    </div>
  );
}

// Create the main page component as a Server Component
export default async function Page({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const p = await searchParams;
  const query = p.q;
  const language = p.lang || 'all';
  let lang: Language;
  switch (language) {
    case 'en':
    case 'ar':
    case 'ur':
      lang = language;
      break;
    default:
      lang = 'all';
  }

  const page = parseInt(p.page || '1');

  // Get both search results and stats in parallel if there's a query
  const [searchData, stats] = query
    ? await Promise.all([
      performSearch(query, page, lang),
      getStats()
    ])
    : [null, await getStats()];

  const baseUrl = `/enhanced?q=${encodeURIComponent(query || '')}&lang=${lang}`;


  // Dynamic metadata based on search query
  metadata.title = query
    ? `Salafi Search Enhanced | ${query}`
    : 'Salafi Search Enhanced';

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <SearchForm
          query={query}
          language={language}
        />


        {query && searchData && (
          <>
            <SearchResults
              results={searchData.results}
              stats={stats}
              queryTime={searchData.query_time_ms}
            />
            <Pagination
              currentPage={searchData.currentPage}
              totalPages={searchData.totalPages}
              baseUrl={baseUrl}
            />
          </>
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

// Error boundary component
export function ErrorBoundary({
  error,
}: {
  error: Error;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-gray-50 to-white p-6">
      <Card className="w-full max-w-lg">
        <CardContent className="p-6">
          <h2 className="text-xl font-semibold text-red-600 mb-4">Error</h2>
          <p className="text-gray-600">{error.message}</p>
          <Button
            onClick={() => window.location.href = '/'}
            className="mt-4"
          >
            Return to Search
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

// Loading state component
export function Loading() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-gray-50 to-white p-6">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
        <p className="mt-4 text-gray-600">Loading search results...</p>
      </div>
    </div>
  );
}