package activation

import "gonum.org/v1/gonum/mat"

type IActivation interface {
	Predictions(outputs *mat.Dense) *mat.Dense
}
