"use client"

import { type FC, useState } from 'react'
import { Search } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from '@/components/ui/alert'

interface SearchResult {
  url: string
  score: number
  title: string
  content: string
  original_language: string
  snippet?: string
}

interface SearchResponse {
  results: SearchResult[]
  total_hits: number
  query_time_ms: number
}

interface SearchStats {
  document_count: number
  index_size_bytes: number
  languages: string[]
  creation_date: string
}

const SearchApp: FC = () => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<SearchStats | null>(null)

  const handleSearch = async (): Promise<void> => {
    if (!query.trim()) return
    
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          size: 10,
          min_score: 0.5,
        }),
      })

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`)
      }

      const data = await response.json() as SearchResponse
      setResults(data.results)
      
      // Fetch stats after successful search
      const statsResponse = await fetch('/api/stats')
      const statsData = await statsResponse.json() as SearchStats
      setStats(statsData)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Cross-Language Search</h1>
          <p className="text-gray-600">Search Islamic content in any language</p>
        </div>

        {/* Search Controls */}
        <Card>
          <CardHeader>
            <CardTitle>Search</CardTitle>
            <CardDescription>
              Enter your search query in any language
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Input
                placeholder="Type your search query..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyUp={handleKeyPress}
                className="flex-1"
              />
              <Button 
                onClick={handleSearch}
                disabled={loading || !query.trim()}
              >
                <Search className="w-4 h-4 mr-2" />
                {loading ? 'Searching...' : 'Search'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Results */}
        {results.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Search Results</CardTitle>
              <CardDescription>
                Found {results.length} matching documents
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {results.map((result, index) => (
                  <div key={index} className="space-y-2 pb-4 border-b last:border-0">
                    <div className="flex justify-between items-start">
                      <a 
                        href={result.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:underline font-medium"
                      >
                        {result.title}
                      </a>
                      <span className="text-sm text-gray-500">
                        Score: {result.score.toFixed(2)}
                      </span>
                    </div>
                    {result.snippet ? (
                      <p className="text-gray-600">{result.snippet}</p>
                    ) : (
                      <p className="text-gray-600">{result.content?.substring(0, 200)}...</p>
                    )}
                    <p className="text-sm text-gray-500">
                      Original language: {result.original_language.toUpperCase()}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Stats */}
        {stats && (
          <Card>
            <CardHeader>
              <CardTitle>Index Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Total Documents</p>
                  <p className="text-2xl font-semibold">{stats.document_count}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Index Size</p>
                  <p className="text-2xl font-semibold">
                    {(stats.index_size_bytes / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

export default SearchApp