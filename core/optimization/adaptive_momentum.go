package optimization

import (
	"github.com/saent-x/ids-nn/core"
	"github.com/saent-x/ids-nn/core/layer"
	"gonum.org/v1/gonum/mat"
	"math"
)

type AdaptiveMomentum struct {
	Optimizer
	Epsilon float64
	Beta_1  float64
	Beta_2  float64
}

func CreateAdaptiveMomentum(learning_rate float64, decay float64, epsilon float64, beta_1 float64, beta_2 float64) *AdaptiveMomentum {

	adam := new(AdaptiveMomentum)

	adam.LearningRate = learning_rate
	adam.CurrentLearningRate = learning_rate
	adam.Decay = decay
	adam.Iterations = 0.0
	adam.Epsilon = epsilon
	adam.Beta_1 = beta_1
	adam.Beta_2 = beta_2

	return adam
}

func (self *AdaptiveMomentum) PreUpdateParams() {
	if self.Decay != 0 {
		self.CurrentLearningRate = self.LearningRate * (1. / (1. + self.Decay*self.Iterations))
	}
}

func (self *AdaptiveMomentum) UpdateParams(layer *layer.Layer) {
	core.ContainsNaN(layer.D_Weights)
	var new_weights, new_biases mat.Dense

	if layer.Weights_Cache == nil || layer.Biases_Cache == nil {
		layer.Weights_Cache = mat.DenseCopyOf(layer.Weights)
		layer.Biases_Cache = mat.DenseCopyOf(layer.Biases)
		layer.Weights_Momentum = mat.DenseCopyOf(layer.Weights)
		layer.Biases_Momentum = mat.DenseCopyOf(layer.Biases)

		layer.Weights_Cache.Zero()
		layer.Biases_Cache.Zero()
		layer.Weights_Momentum.Zero()
		layer.Biases_Momentum.Zero()
	}

	var dweights_beta, dbiases_beta, weights_momentum_beta, bias_momentum_beta mat.Dense

	dweights_beta.Apply(func(i, j int, v float64) float64 {
		return (1 - self.Beta_1) * v
	}, layer.D_Weights)
	dbiases_beta.Apply(func(i, j int, v float64) float64 {
		return (1 - self.Beta_1) * v
	}, layer.D_Biases)

	weights_momentum_beta.Apply(func(i, j int, v float64) float64 {
		return self.Beta_1 * v
	}, layer.Weights_Momentum)
	bias_momentum_beta.Apply(func(i, j int, v float64) float64 {
		return self.Beta_1 * v
	}, layer.Biases_Momentum)

	layer.Weights_Momentum.Add(&weights_momentum_beta, &dweights_beta)
	layer.Biases_Momentum.Add(&bias_momentum_beta, &dbiases_beta)

	// corrected momentums
	var weight_momentums_corrected, bias_momentum_corrected mat.Dense
	weight_momentums_corrected.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(self.Beta_1, self.Iterations+1))
	}, layer.Weights_Momentum)
	bias_momentum_corrected.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(self.Beta_1, self.Iterations+1))
	}, layer.Biases_Momentum)

	var dweights_sqr_beta, dbiases_sqr_beta, weight_cache_beta_2, bias_cache_beta_2 mat.Dense

	dweights_sqr_beta.Apply(func(i, j int, v float64) float64 {
		return (1 - self.Beta_2) * math.Pow(v, 2)
	}, layer.D_Weights)
	dbiases_sqr_beta.Apply(func(i, j int, v float64) float64 {
		return (1 - self.Beta_2) * math.Pow(v, 2)
	}, layer.D_Biases)

	weight_cache_beta_2.Apply(func(i, j int, v float64) float64 {
		return self.Beta_2 * v
	}, layer.Weights_Cache)
	bias_cache_beta_2.Apply(func(i, j int, v float64) float64 {
		return self.Beta_2 * v
	}, layer.Biases_Cache)

	layer.Weights_Cache.Add(&weight_cache_beta_2, &dweights_sqr_beta)
	layer.Biases_Cache.Add(&bias_cache_beta_2, &dbiases_sqr_beta)

	// Corrected Cache
	var weights_cache_corrected, bias_cache_corrected mat.Dense
	weights_cache_corrected.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(self.Beta_2, self.Iterations+1))
	}, layer.Weights_Cache)
	bias_cache_corrected.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(self.Beta_2, self.Iterations+1))
	}, layer.Biases_Cache)

	// Vanilla SGD
	new_weights.Apply(func(i, j int, v float64) float64 {
		return -self.CurrentLearningRate * v
	}, &weight_momentums_corrected)
	new_biases.Apply(func(i, j int, v float64) float64 {
		return -self.CurrentLearningRate * v
	}, &bias_momentum_corrected)

	var weights_epsilon_sum, biases_epsilon_sum mat.Dense

	weights_epsilon_sum.Apply(func(i, j int, v float64) float64 {
		return math.Sqrt(v) + self.Epsilon
	}, &weights_cache_corrected)
	biases_epsilon_sum.Apply(func(i, j int, v float64) float64 {
		return math.Sqrt(v) + self.Epsilon
	}, &bias_cache_corrected)

	var weights_div, biases_div mat.Dense

	weights_div.DivElem(&new_weights, &weights_epsilon_sum)
	biases_div.DivElem(&new_biases, &biases_epsilon_sum)

	var weights_sum, biases_sum mat.Dense

	weights_sum.Add(layer.Weights, &weights_div)
	biases_sum.Add(layer.Biases, &biases_div)

	layer.Weights = mat.DenseCopyOf(&weights_sum)
	layer.Biases = mat.DenseCopyOf(&biases_sum)
}

func (self *AdaptiveMomentum) PostUpdateParams() {
	self.Iterations += 1
}
