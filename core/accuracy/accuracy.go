package accuracy

import (
	"gonum.org/v1/gonum/mat"
)

// Accuracy describes how often the largest confidence is the correct class in terms of fraction
type IAccuracy interface {
	Init(y *mat.Dense, reinit bool)
	Calculate(outputs *mat.Dense, y *mat.Dense) float64
	Compare(predictions, y *mat.Dense) *mat.Dense
	NewPass()
	CalculateAccumulated() float64
}

type Accuracy struct {
	AccumulatedSum   float64
	AccumulatedCount float64
}

func (accuracy *Accuracy) CalculateAccumulated() float64 {
	dataLoss := accuracy.AccumulatedSum / accuracy.AccumulatedCount

	return dataLoss
}

func (accuracy *Accuracy) NewPass() {
	accuracy.AccumulatedSum = 0
	accuracy.AccumulatedCount = 0
}
