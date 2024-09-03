package core

import (
	"testing"
)

func TestSpiralData(t *testing.T) {
	X, y := SpiralData(100, 3)

	if X == nil || len(y) == 0 {
		t.Errorf("error: X & y spiral data is empty")
	}
}

func TestScatterPlotFunction(t *testing.T) {
	X, _ := SpiralData(100, 3)

	result := PlotData(X, 100, 3)

	if !result {
		t.Errorf("error: no plot was made")
	}
}
