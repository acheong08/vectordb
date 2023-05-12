package bbolt

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"go.etcd.io/bbolt"
)

type Database struct {
	db        *bbolt.DB
	increment int
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

		for _, embedding := range embeddings {
			// Convert the float64 slice to a byte slice
			data := make([]byte, len(embedding)*8)
			for i, v := range embedding {
				binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(v))
			}
			err = bucket.Put([]byte(fmt.Sprint(b.increment)), data)
			if err != nil {
				return err
			}
			b.increment++
		}
		return nil
	})
}

func (b *Database) Load() ([][]float64, error) {
	var embeddings [][]float64

	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("embeddings"))
		if bucket == nil {
			return errors.New("bucket not found")
		}

		return bucket.ForEach(func(k, v []byte) error {
			var embedding []float64
			for i := 0; i < len(v); i += 8 {
				embedding = append(embedding, math.Float64frombits(binary.LittleEndian.Uint64(v[i:])))
			}
			embeddings = append(embeddings, embedding)
			return nil
		})
	})
	return embeddings, err
}

func (b *Database) Add(embedding []float64) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket, err := tx.CreateBucketIfNotExists([]byte("embeddings"))
		if err != nil {
			return err
		}

		// Convert the float64 slice to a byte slice
		data := make([]byte, len(embedding)*8)
		for i, v := range embedding {
			binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(v))
		}
		err = bucket.Put([]byte(fmt.Sprint(b.increment)), data)
		if err != nil {
			return err
		}
		b.increment++
		return nil
	})
}

func (b *Database) Close() error {
	return b.db.Close()
}

func (b *Database) Delete() error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		return tx.DeleteBucket([]byte("embeddings"))
	})
}

func (b *Database) Remove(key string) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("embeddings"))
		if bucket == nil {
			return errors.New("bucket not found")
		}
		return bucket.Delete([]byte(key))
	})
}
