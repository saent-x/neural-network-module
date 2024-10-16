package layer

import (
	"github.com/saent-x/ids-nn/core/datamodels"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

type IDenseLayer interface {
	Forward(inputs *mat.Dense, training bool)
	Backward(d_values *mat.Dense)
	GetOutput() *mat.Dense
	GetDInputs() *mat.Dense
	SetDInputs(inputs *mat.Dense)
	Reset()
}

type ILayerNavigation interface {
	GetPreviousLayer() interface{}
	GetNextLayer() interface{}
	SetPreviousLayer(prev interface{})
	SetNextLayer(next interface{})
}

// this interface abstracts all layers and activations
type ILayer interface {
	IDenseLayer
	ILayerNavigation
}

type Layer struct {
	Weights *mat.Dense
	Biases  *mat.Dense

	Weights_Momentum *mat.Dense
	Biases_Momentum  *mat.Dense

	Weights_Cache *mat.Dense
	Biases_Cache  *mat.Dense

	D_Weights *mat.Dense
	D_Biases  *mat.Dense

	Weight_Regularizer_L1 float64
	Weight_Regularizer_L2 float64
	Biases_Regularizer_L1 float64
	Biases_Regularizer_L2 float64

	LayerCommons
	LayerNavigation
}

func CreateLayer(n_inputs int, n_neurons int, weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2 float64) *Layer {
	layer := new(Layer)

	layer.Biases = mat.NewDense(1, n_neurons, make([]float64, n_neurons))
	layer.Biases.Zero()
	layer.Weights = mat.NewDense(n_inputs, n_neurons, nil)

	layer.Weights.Apply(func(i, j int, v float64) float64 {
		return 0.1 * rand.NormFloat64()
	}, layer.Weights)

	layer.Weight_Regularizer_L1 = weight_regularizer_l1
	layer.Weight_Regularizer_L2 = weight_regularizer_l2

	layer.Biases_Regularizer_L1 = bias_regularizer_l1
	layer.Biases_Regularizer_L2 = bias_regularizer_l2

	return layer
}

func (layer *Layer) Forward(inputs *mat.Dense, training bool) {
	layer.Inputs = mat.DenseCopyOf(inputs) // set inputs to be used for backpropagation

	// calculate dot product between inputs and weights and store in output var.
	var output mat.Dense
	output.Mul(inputs, layer.Weights)

	// adds a bias to each row of the resulting dot product
	output.Apply(func(i, j int, value float64) float64 {
		return value + layer.Biases.At(0, j)
	}, &output)

	layer.Output = mat.DenseCopyOf(&output)
}

func (layer *Layer) Backward(d_values *mat.Dense) {
	_, c := d_values.Dims()
	inputs_T := layer.Inputs.T()
	r0, _ := inputs_T.Dims()

	// Gradients on parameter - dot product between inputs and d_values
	layer.D_Weights = mat.NewDense(r0, c, nil)
	layer.D_Weights.Mul(inputs_T, d_values)

	// sum all cols in kd_values
	//col-wise and retain dims
	layer.D_Biases = mat.NewDense(1, c, nil)
	for i := 0; i < c; i++ {
		layer.D_Biases.SetCol(i, []float64{mat.Sum(d_values.ColView(i))})
	}

	if layer.Weight_Regularizer_L1 > 0 {
		d_l1 := mat.DenseCopyOf(layer.Weights)
		d_l1.Apply(func(i, j int, v float64) float64 {
			return lo.Ternary(v < 0, -1., 1.)
		}, layer.Weights)

		var new_dweights mat.Dense
		new_dweights.Apply(func(i, j int, v float64) float64 {
			return layer.Weight_Regularizer_L1 * v
		}, d_l1)

		layer.D_Weights.Add(layer.D_Weights, &new_dweights)
	}

	if layer.Weight_Regularizer_L2 > 0 {
		var new_dweights mat.Dense
		new_dweights.Apply(func(i, j int, v float64) float64 {
			return (2 * layer.Weight_Regularizer_L2) * v
		}, layer.Weights)

		layer.D_Weights.Add(layer.D_Weights, &new_dweights)
	}

	if layer.Biases_Regularizer_L1 > 0 {
		d_l1 := mat.DenseCopyOf(layer.Biases)
		d_l1.Apply(func(i, j int, v float64) float64 {
			return lo.Ternary(v < 0, -1., 1.)
		}, layer.Biases)

		var new_dbiases mat.Dense
		new_dbiases.Apply(func(i, j int, v float64) float64 {
			return layer.Biases_Regularizer_L1 * v
		}, d_l1)

		layer.D_Biases.Add(layer.D_Biases, &new_dbiases)
	}

	if layer.Biases_Regularizer_L2 > 0 {
		var new_dbiases mat.Dense
		new_dbiases.Apply(func(i, j int, v float64) float64 {
			return (2 * layer.Biases_Regularizer_L2) * v
		}, layer.Biases)

		layer.D_Biases.Add(layer.D_Biases, &new_dbiases)
	}

	var result mat.Dense
	result.Mul(d_values, layer.Weights.T())

	layer.D_Inputs = mat.DenseCopyOf(&result)
}

func (layer *Layer) GetParameters() datamodels.ModelParameter {
	return datamodels.ModelParameter{layer.Weights, layer.Biases}
}

func (layer *Layer) SetParameters(parameter datamodels.ModelParameter) {
	layer.Weights = mat.DenseCopyOf(parameter.Weights)
	layer.Biases = mat.DenseCopyOf(parameter.Biases)
}
