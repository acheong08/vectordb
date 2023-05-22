package experimental

type Chunk struct {
	Documents  []Document
	Embeddings []float64
}

func (db *Chunk) GetEmbeddings() []float64 {
	return db.Embeddings
}

func (db *Chunk) GetDocumentEmbeddings() [][]float64 {
	embeddings := make([][]float64, len(db.Documents))
	for i, doc := range db.Documents {
		embeddings[i] = doc.GetEmbeddings()
	}
	return embeddings
}

func (db *Chunk) RemoveByIndex(i int) {
	db.Documents = append(db.Documents[:i], db.Documents[i+1:]...)
}

func (db *Chunk) RemoveByText(text string) {
	for i, doc := range db.Documents {
		if doc.Text == text {
			db.RemoveByIndex(i)
			return
		}
	}
}

func (db *Chunk) GetTextByIndex(i int) string {
	return db.Documents[i].Text
}

func (db *Chunk) GetEmbeddingByIndex(i int) []float64 {
	return db.Documents[i].Embeddings
}

func (db *Chunk) GetTexts() []string {
	texts := make([]string, len(db.Documents))
	for i, doc := range db.Documents {
		texts[i] = doc.Text
	}
	return texts
}
func (Chunk) IsUnit() {}

func (Chunk) GetType() string {
	return "chunk"
}
