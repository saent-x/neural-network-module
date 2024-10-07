package datawrappers

type ModelWrapper struct {
	Layers    []LayerWrapper
	Loss      string `json:"loss,omitempty"`
	Accuracy  string `json:"accuracy,omitempty"`
	Optimizer OptimizerWrapper
}

type LayerWrapper struct {
	Type    string
	Weights MatDenseWrapper `json:"weights"`
	Biases  MatDenseWrapper `json:"biases"`

	Weights_Momentum MatDenseWrapper `json:"weights___momentum"`
	Biases_Momentum  MatDenseWrapper `json:"biases___momentum"`

	Weights_Cache MatDenseWrapper `json:"weights___cache"`
	Biases_Cache  MatDenseWrapper `json:"biases___cache"`

	D_Weights MatDenseWrapper `json:"d___weights"`
	D_Biases  MatDenseWrapper `json:"d___biases"`

	Weight_Regularizer_L1 float64 `json:"weight___regularizer___l_1,omitempty"`
	Weight_Regularizer_L2 float64 `json:"weight___regularizer___l_2,omitempty"`
	Biases_Regularizer_L1 float64 `json:"biases___regularizer___l_1,omitempty"`
	Biases_Regularizer_L2 float64 `json:"biases___regularizer___l_2,omitempty"`
}

type MatDenseWrapper struct {
	Data []float64 `json:"data,omitempty"`
	Rows int       `json:"rows,omitempty"`
	Cols int       `json:"cols,omitempty"`
}

type OptimizerWrapper struct {
	Type string
	Obj  interface{}
}
