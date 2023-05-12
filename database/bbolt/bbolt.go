package bbolt

import (
	"encoding/json"
	"errors"

	"go.etcd.io/bbolt"
)

type Database struct {
	db *bbolt.DB
}

func New(filePath string) (*Database, error) {
	db, err := bbolt.Open(filePath, 0600, nil)
	if err != nil {
		return nil, err
	}
	return &Database{db: db}, nil
}

func (b *Database) Store(embeddings [][]float64) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket, err := tx.CreateBucketIfNotExists([]byte("embeddings"))
		if err != nil {
			return err
		}

		data, err := json.Marshal(embeddings)
		if err != nil {
			return err
		}

		return bucket.Put([]byte("embeddings"), data)
	})
}

func (b *Database) Load() ([][]float64, error) {
	var embeddings [][]float64

	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("embeddings"))
		if bucket == nil {
			return errors.New("Bucket not found")
		}

		data := bucket.Get([]byte("embeddings"))
		return json.Unmarshal(data, &embeddings)
	})

	return embeddings, err
}

func (b *Database) Add(embedding []float64) error {
	embeddings, err := b.Load()
	if err != nil {
		return err
	}

	embeddings = append(embeddings, embedding)

	return b.Store(embeddings)
}
