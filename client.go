package vectordb

import (
	"github.com/acheong08/vectordb/database"
	"github.com/acheong08/vectordb/rank"
	"github.com/acheong08/vectordb/typings"
	"github.com/acheong08/vectordb/vectors"
)

type Client struct {
	DB *database.Database
}

func NewClient(filePath string) (*Client, error) {
	db, err := database.New(filePath)
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
	c.DB.Add(database.Document{
		Text:      text,
		Embedding: embedding,
	})
	return nil
}

func (c *Client) Query(queries []string, topK int, sorted bool) ([][]typings.SearchResult, error) {
	embeddings, err := vectors.EncodeMulti(queries)
	if err != nil {
		return nil, err
	}
	return rank.Rank(embeddings, c.DB.GetEmbeddings(), topK, sorted), nil
}

func (c *Client) RemoveByText(text string) error {
	c.DB.RemoveByText(text)
	return nil
}

func (c *Client) RemoveByIndex(i int) error {
	c.DB.RemoveByIndex(i)
	return nil
}

func (c *Client) Save() error {
	return c.DB.Save()
}
