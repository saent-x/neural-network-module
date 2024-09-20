package optimization

import (
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
	"math"
)

type RootMeanSquarePropagation struct {
	LearningRate        float64
	CurrentLearningRate float64
	Decay               float64
	Iterations          float64
	Epsilon             float64
	Rho                 float64
}

func CreateRootMeanSquarePropagation(learning_rate float64, decay float64, epsilon float64, rho float64) *RootMeanSquarePropagation {
	return &RootMeanSquarePropagation{
		LearningRate:        learning_rate,
		CurrentLearningRate: learning_rate,
		Decay:               decay,
		Iterations:          0.0,
		Epsilon:             epsilon,
		Rho:                 rho,
	}
}

func (self *RootMeanSquarePropagation) PreUpdateParams() {
	if self.Decay != 0 {
		self.CurrentLearningRate = self.LearningRate * (1. / (1. + self.Decay*self.Iterations))
	}
}

func (self *RootMeanSquarePropagation) UpdateParams(layer *layer.Layer) {
	r, c := layer.D_Weights.Dims()
	r0, c0 := layer.D_Biases.Dims()

	new_weights := mat.NewDense(r, c, nil)
	new_biases := mat.NewDense(r0, c0, nil)

	if layer.Weights_Cache == nil || layer.Biases_Cache == nil {
		layer.Weights_Cache = mat.DenseCopyOf(layer.Weights)
		layer.Biases_Cache = mat.DenseCopyOf(layer.Biases)

		layer.Weights_Cache.Zero()
		layer.Biases_Cache.Zero()
	}

	var dweights_sqr_rho, dbiases_sqr_rho, weights_cache_rho, bias_cache_rho mat.Dense

	dweights_sqr_rho.Apply(func(i, j int, v float64) float64 {
		return (1 - self.Rho) * math.Pow(v, 2)
	}, layer.D_Weights)
	dbiases_sqr_rho.Apply(func(i, j int, v float64) float64 {
		return (1 - self.Rho) * math.Pow(v, 2)
	}, layer.D_Biases)

	weights_cache_rho.Apply(func(i, j int, v float64) float64 {
		return self.Rho * v
	}, layer.Weights_Cache)
	bias_cache_rho.Apply(func(i, j int, v float64) float64 {
		return self.Rho * v
	}, layer.Biases_Cache)

	layer.Weights_Cache.Add(&weights_cache_rho, &dweights_sqr_rho)
	layer.Biases_Cache.Add(&bias_cache_rho, &dbiases_sqr_rho)

	new_weights.Apply(func(i, j int, v float64) float64 {
		return -self.CurrentLearningRate * v
	}, layer.D_Weights)
	new_biases.Apply(func(i, j int, v float64) float64 {
		return -self.CurrentLearningRate * v
	}, layer.D_Biases)

	var weights_cache_sqrt, biases_cache_sqr, weights_epsilon_sum, biases_epsilon_sum mat.Dense

	weights_cache_sqrt.Apply(func(i, j int, v float64) float64 {
		return math.Sqrt(v)
	}, layer.Weights_Cache)
	biases_cache_sqr.Apply(func(i, j int, v float64) float64 {
		return math.Sqrt(v)
	}, layer.Biases_Cache)

	weights_epsilon_sum.Apply(func(i, j int, v float64) float64 {
		return v + self.Epsilon
	}, &weights_cache_sqrt)
	biases_epsilon_sum.Apply(func(i, j int, v float64) float64 {
		return v + self.Epsilon
	}, &biases_cache_sqr)

	var weights_div, biases_div mat.Dense

	weights_div.DivElem(new_weights, &weights_epsilon_sum)
	biases_div.DivElem(new_biases, &biases_epsilon_sum)

	var weights_sum, biases_sum mat.Dense

	weights_sum.Add(layer.Weights, &weights_div)
	biases_sum.Add(layer.Biases, &biases_div)

	layer.Weights = mat.DenseCopyOf(&weights_sum)
	layer.Biases = mat.DenseCopyOf(&biases_sum)
}

func (self *RootMeanSquarePropagation) PostUpdateParams() {
	self.Iterations += 1
}
