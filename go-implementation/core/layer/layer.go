package layer

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	_weights *mat.Dense
	_biases  *mat.Dense

	Output *mat.Dense
}

func CreateLayer(n_inputs int, n_neurons int) *Layer {
	//r := distuv.Normal{
	//	Mu:    0,
	//	Sigma: 1,
	//	Src:   rand.NewSource(0), // Seed for reproducibility
	//}

	layer := new(Layer)

	layer._biases = mat.NewDense(1, n_neurons, make([]float64, n_neurons))
	layer._biases.Zero()
	layer._weights = mat.NewDense(n_inputs, n_neurons, nil)

	layer._weights.Apply(func(i, j int, v float64) float64 {
		return 0.01 * rand.NormFloat64() //r.Rand()
	}, layer._weights)

	return layer
}

func (layer *Layer) Forward(inputs *mat.Dense) {
	rows, _ := inputs.Dims()

	// calculate dot product between inputs and weights and store in output var.
	output := mat.NewDense(rows, layer._biases.RawMatrix().Cols, nil)
	output.Mul(inputs, layer._weights)

	// adds a bias to each row of the resulting dot product
	output.Apply(func(i, j int, value float64) float64 {
		return value + layer._biases.At(0, j)
	}, output)

	layer.Output = output
}
