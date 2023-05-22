package experimental

type Document struct {
	Text       string    `json:"text"`
	Embeddings []float64 `json:"embeddings"`
}

// Type assertions
func (Document) IsUnit() {}

func (Document) GetType() string {
	return "document"
}

func (doc *Document) GetEmbeddings() []float64 {
	return doc.Embeddings
}
