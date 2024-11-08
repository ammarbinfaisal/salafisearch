"use client";

import { type FC, useState } from 'react';
import { Search, Sparkles } from 'lucide-react';
import {
  Card,
  CardContent,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { Badge } from "@/components/ui/badge";

interface SearchResult {
  url: string;
  score: number;
  title: {
    text: string;
    language: string;
  };
  content: {
    text: string;
    language: string;
  };
  original_language: string;
  available_translations: string[];
  domain: string;
  timestamp: number;
}

interface SearchResponse {
  results: SearchResult[];
  total: number;
  query_time_ms: number;
  languages: string[];
}

interface SearchStats {
  document_count: number;
  index_size_bytes: number;
  languages: string[];
  creation_date: string;
}

const LANGUAGE_NAMES: Record<string, string> = {
  en: "English",
  ar: "Arabic"
};

const SearchApp: FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<SearchStats | null>(null);
  const [matchPhrase, setMatchPhrase] = useState(true);
  const [queryTime, setQueryTime] = useState<number | null>(null);

  const handleSearch = async (): Promise<void> => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResults([]);
    setQueryTime(null);

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          limit: 15,
          titleWeight: 1.5,
          contentWeight: 1.2,
          matchPhrase,
          cache: true
        }),
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json() as SearchResponse;
      setResults(data.results);
      setQueryTime(data.query_time_ms);

      try {
        const statsResponse = await fetch('/api/stats');
        if (statsResponse.ok) {
          const statsData = await statsResponse.json() as SearchStats;
          setStats(statsData);
        }
      } catch (statsErr) {
        console.error('Failed to fetch stats:', statsErr);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString();
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Search Bar */}
        <div className="relative">
          <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-gray-400" />
          </div>
          <Input
            placeholder="Search across languages..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyUp={handleKeyPress}
            className="pl-10 h-12 text-lg shadow-sm"
          />
          <Button 
            onClick={handleSearch} 
            disabled={loading || !query.trim()}
            className="absolute right-1.5 top-1.5 h-9"
          >
            {loading ? (
              <div className="flex items-center">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                Searching
              </div>
            ) : (
              <div className="flex items-center">
                <Sparkles className="w-4 h-4 mr-2" />
                Search
              </div>
            )}
          </Button>
        </div>

        {/* Search Controls */}
        <div className="flex flex-wrap items-center gap-4 px-1">
          <div className="flex items-center space-x-2">
            <Switch
              checked={matchPhrase}
              onCheckedChange={setMatchPhrase}
              id="phrase-mode"
              className="data-[state=checked]:bg-blue-600"
            />
            <label htmlFor="phrase-mode" className="text-sm text-gray-600">
              Exact phrase
            </label>
          </div>

          {queryTime && (
            <span className="text-sm text-gray-500 ml-auto">
              {queryTime}ms
            </span>
          )}
        </div>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive" className="animate-in fade-in">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Results */}
        {results.length > 0 ? (
          <div className="space-y-4">
            {results.map((result, index) => (
              <Card key={index} className="overflow-hidden hover:shadow-md transition-shadow">
                <CardContent className="p-6">
                  <div className="space-y-3">
                    <div className="flex justify-between items-start gap-4">
                      <div className="space-y-1 min-w-0">
                        <a
                          href={result.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 font-medium block truncate"
                          dangerouslySetInnerHTML={{ __html: result.title.text }}
                        />
                        <div className="flex gap-2 items-center flex-wrap">
                          <Badge variant="secondary" className="bg-blue-50 text-blue-700">
                            {LANGUAGE_NAMES[result.title.language] || result.title.language}
                          </Badge>
                          <span className="text-sm text-gray-500">
                            {result.domain} • {formatDate(result.timestamp)}
                          </span>
                        </div>
                      </div>
                      <Badge variant="outline" className="shrink-0">
                        {result.score.toFixed(2)}
                      </Badge>
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
          !loading &&
          query &&
          !error && (
            <div className="text-center py-12 text-gray-500">
              No results found for your search query.
            </div>
          )
        )}

        {/* Stats */}
        {stats && (
          <div className="flex gap-6 justify-center text-sm text-gray-500 pt-4">
            <span>{stats.document_count.toLocaleString()} documents indexed</span>
            <span>•</span>
            <span>{(stats.index_size_bytes / (1024 * 1024)).toFixed(2)} MB index size</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchApp;