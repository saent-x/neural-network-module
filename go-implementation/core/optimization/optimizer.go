package optimization

import "github.com/saent-x/ids-nn/core/layer"

type IOptimizer interface {
	PreUpdateParams()
	UpdateParams(layer *layer.Layer)
	PostUpdateParams()
	GetCurrentLearningRate() float64
}

type Optimizer struct {
	LearningRate        float64
	CurrentLearningRate float64
	Decay               float64
	Iterations          float64
}

func (o *Optimizer) GetCurrentLearningRate() float64 {
	return o.CurrentLearningRate
}
