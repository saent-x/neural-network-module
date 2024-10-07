package datamodels

import "gonum.org/v1/gonum/mat"

type ValidationData struct {
	X, Y *mat.Dense
}

type TrainingData struct {
	X, Y *mat.Dense
}
