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

func TestPlotForSineData(t *testing.T) {
	X, y := SineData(1000)
	result := PlotSineData(X, y)

	if !result {
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

	result := PlotScatter(X, 100, 3, "spiral")

	if !result {
		t.Errorf("error: no plot was made")
	}
}

func TestScatterPlotFunctionForVerticalData(t *testing.T) {
	X, _ := VerticalData(100, 3)

	result := PlotScatter(X, 100, 3, "vertical")

	if !result {
		t.Errorf("error: no plot was made")
	}
}

func TestSparseToOHE_1(t *testing.T) {
	data := mat.NewDense(1, 3, []float64{0, 0, 1})
	result := SparseToOHE(data, 3)

	fmt.Println(mat.Formatted(result))
}
