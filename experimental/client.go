package experimental

import (
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
func (c *Client) Query(queries []string, topK int) ([][]typings.SearchResult, [][]*Document, error) {
	embeddings, err := vectors.EncodeMulti(queries)
	if err != nil {
		return nil, nil, err
	}

	allSearchResults := make([][]typings.SearchResult, len(queries))
	allDocuments := make([][]*Document, len(queries))

	for qIndex, queryEmbedding := range embeddings {
		pending_units := make([]*Chunk, 0, topK)
		true_documents := make([]*Document, 0, topK)

		for _, result := range rank.Rank([][]float64{queryEmbedding}, c.DB.GetEmbeddings(), topK, true)[0] {
			unit := c.DB.GetUnitByIndex(result.CorpusID)
			if unit.GetType() == "chunk" {
				pending_units = append(pending_units, unit.(*Chunk))
			} else {
				true_documents = append(true_documents, unit.(*Document))
			}
		}

		for len(pending_units) > 0 {
			next_pending_units := make([]*Chunk, 0, len(pending_units))
			no_more_chunks := true
			for _, chunk := range pending_units {
				for _, result := range rank.Rank([][]float64{queryEmbedding}, chunk.GetDocumentEmbeddings(), topK, true)[0] {
					unit := c.DB.GetUnitByIndex(result.CorpusID)
					if unit.GetType() == "chunk" {
						next_pending_units = append(next_pending_units, unit.(*Chunk))
						no_more_chunks = false
					} else {
						true_documents = append(true_documents, unit.(*Document))
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

		ranked_results := rank.Rank([][]float64{queryEmbedding}, true_embeddings, topK, true)[0]

		searchResults := make([]typings.SearchResult, 0, len(ranked_results))
		searchResults = append(searchResults, ranked_results...)

		allSearchResults[qIndex] = searchResults
		allDocuments[qIndex] = true_documents
	}

	return allSearchResults, allDocuments, nil
}
func (c *Client) RemoveByIndex(i int) error {
	c.DB.RemoveByIndex(i)
	return nil
}

func (c *Client) Save() error {
	return c.DB.Save()
}
