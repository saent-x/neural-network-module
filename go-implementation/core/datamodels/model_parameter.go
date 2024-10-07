package datamodels

import "gonum.org/v1/gonum/mat"

type ModelParameter struct {
	Weights *mat.Dense
	Biases  *mat.Dense
}
