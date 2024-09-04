package core

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestSpiralData(t *testing.T) {
	X, y := SpiralData(100, 3)

	fmt.Println(mat.Formatted(X))
	fmt.Println(mat.Formatted(y))

	if X == nil || y == nil {
		t.Errorf("error: X & y spiral data is empty")
	}
}

func TestVerticalData(t *testing.T) {
	X, y := VerticalData(100, 3)

	fmt.Println(mat.Formatted(X))
	fmt.Println(mat.Formatted(y))

	if X == nil || y == nil {
		t.Errorf("error: X & y vertical data is empty")
	}
}

func TestScatterPlotFunctionForSpiralData(t *testing.T) {
	X, _ := SpiralData(100, 3)

	result := PlotData(X, 100, 3, "spiral")

	if !result {
		t.Errorf("error: no plot was made")
	}
}

func TestScatterPlotFunctionForVerticalData(t *testing.T) {
	X, _ := VerticalData(100, 3)

	result := PlotData(X, 100, 3, "vertical")

	if !result {
		t.Errorf("error: no plot was made")
	}
}
