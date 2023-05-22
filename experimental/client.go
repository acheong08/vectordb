package experimental

import (
	"sort"

	"github.com/acheong08/vectordb/rank"
	"github.com/acheong08/vectordb/typings"
	"github.com/acheong08/vectordb/vectors"
)

type Client struct {
	DB *Database
}

func NewClient(filePath string) (*Client, error) {
	db, err := NewDatabase(filePath)
	if err != nil {
		return nil, err
	}
	return &Client{
		DB: db,
	}, nil
}

func (c *Client) Add(text string) error {
	embedding, err := vectors.Encode(text)
	if err != nil {
		return err
	}
	c.DB.Add(&Document{
		Text:       text,
		Embeddings: embedding,
	})
	return nil
}

func (c *Client) Query(queries string, topK int) ([]typings.SearchResult, []*Document, error) {
	embeddings, err := vectors.Encode(queries)
	if err != nil {
		return nil, nil, err
	}

	var pending_units []*Chunk
	var true_documents []*Document

	for _, result := range rank.Rank([][]float64{embeddings}, c.DB.GetEmbeddings(), topK, true)[0] {
		if c.DB.GetUnitByIndex(result.CorpusID).GetType() == "chunk" {
			pending_units = append(pending_units, c.DB.GetUnitByIndex(result.CorpusID).(*Chunk))
		} else {
			true_documents = append(true_documents, c.DB.GetUnitByIndex(result.CorpusID).(*Document))
		}
	}

	for len(pending_units) > 0 {
		no_more_chunks := true
		next_pending_units := []*Chunk{}
		for _, chunk := range pending_units {
			for _, result := range rank.Rank([][]float64{embeddings}, chunk.GetDocumentEmbeddings(), topK, true)[0] {
				if c.DB.GetUnitByIndex(result.CorpusID).GetType() == "chunk" {
					chunk := c.DB.GetUnitByIndex(result.CorpusID).(*Chunk)
					next_pending_units = append(next_pending_units, chunk)
					no_more_chunks = false
				} else {
					true_documents = append(true_documents, c.DB.GetUnitByIndex(result.CorpusID).(*Document))
				}
			}
		}
		if no_more_chunks {
			break
		}
		pending_units = next_pending_units
	}

	true_embeddings := make([][]float64, len(true_documents))
	for i, doc := range true_documents {
		true_embeddings[i] = doc.GetEmbeddings()
	}

	ranked_results := rank.Rank([][]float64{embeddings}, true_embeddings, topK, true)[0]

	var searchResults []typings.SearchResult
	searchResults = append(searchResults, ranked_results...)

	// Sort the search results
	sort.Slice(searchResults, func(i, j int) bool {
		return searchResults[i].Score > searchResults[j].Score
	})

	return searchResults, true_documents, nil
}

func (c *Client) RemoveByIndex(i int) error {
	c.DB.RemoveByIndex(i)
	return nil
}

func (c *Client) Save() error {
	return c.DB.Save()
}
