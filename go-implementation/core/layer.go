package core

import (
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	_weights mat.Dense
	_biases  mat.Dense
	_output  mat.Dense

	Output *mat.Dense
}

func CreateLayer(n_inputs int32, n_neurons int16) *Layer {
	layer := new(Layer)

	//layer._weights = rand.NormFloat64(1, 2)
	//layer._biases = biases

	return layer
}

func (layer *Layer) Forward(inputs mat.Dense) {
	_ = mat.NewDense(1, 1, nil)

	//layer._output = result.Product(inputs, layer._weights) + layer._biases
}
