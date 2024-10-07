package optimization

import (
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
	"math"
)

type AdaptiveGradient struct {
	Optimizer
	Epsilon float64
}

func CreateAdaptiveGradient(learningRate float64, decay float64, epsilon float64) *AdaptiveGradient {
	ada_grad := new(AdaptiveGradient)

	ada_grad.LearningRate = learningRate
	ada_grad.CurrentLearningRate = learningRate
	ada_grad.Decay = decay
	ada_grad.Iterations = 0.0
	ada_grad.Epsilon = epsilon

	return ada_grad
}

func (self *AdaptiveGradient) PreUpdateParams() {
	if self.Decay != 0 {
		self.CurrentLearningRate = self.LearningRate * (1. / (1. + self.Decay*self.Iterations))
	}
}

func (self *AdaptiveGradient) UpdateParams(layer *layer.Layer) {
	var new_weights, new_biases mat.Dense

	if layer.Weights_Cache == nil || layer.Biases_Cache == nil {
		layer.Weights_Cache = mat.DenseCopyOf(layer.Weights)
		layer.Biases_Cache = mat.DenseCopyOf(layer.Biases)

		layer.Weights_Cache.Zero()
		layer.Biases_Cache.Zero()
	}

	var dweights_sqr, dbiases_sqr mat.Dense

	dweights_sqr.Apply(func(i, j int, v float64) float64 {
		return math.Pow(v, 2)
	}, layer.D_Weights)
	dbiases_sqr.Apply(func(i, j int, v float64) float64 {
		return math.Pow(v, 2)
	}, layer.D_Biases)

	layer.Weights_Cache.Add(layer.Weights_Cache, &dweights_sqr)
	layer.Biases_Cache.Add(layer.Biases_Cache, &dbiases_sqr)

	new_weights.Apply(func(i, j int, v float64) float64 {
		return -self.CurrentLearningRate * v
	}, layer.D_Weights)
	new_biases.Apply(func(i, j int, v float64) float64 {
		return -self.CurrentLearningRate * v
	}, layer.D_Biases)

	var weights_epsilon_sum, biases_epsilon_sum mat.Dense

	weights_epsilon_sum.Apply(func(i, j int, v float64) float64 {
		return math.Sqrt(v) + self.Epsilon
	}, layer.Weights_Cache)
	biases_epsilon_sum.Apply(func(i, j int, v float64) float64 {
		return math.Sqrt(v) + self.Epsilon
	}, layer.Biases_Cache)

	var weights_div, biases_div mat.Dense

	weights_div.DivElem(&new_weights, &weights_epsilon_sum)
	biases_div.DivElem(&new_biases, &biases_epsilon_sum)

	var weights_sum, biases_sum mat.Dense

	weights_sum.Add(layer.Weights, &weights_div)
	biases_sum.Add(layer.Biases, &biases_div)

	layer.Weights = mat.DenseCopyOf(&weights_sum)
	layer.Biases = mat.DenseCopyOf(&biases_sum)
}

func (self *AdaptiveGradient) PostUpdateParams() {
	self.Iterations += 1
}
