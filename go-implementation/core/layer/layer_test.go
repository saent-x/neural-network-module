package layer

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestLayerCreation(t *testing.T) {
	X := mat.NewDense(4, 3, []float64{0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08, 0.02, 0.9, 0.08})

	layer_1 := CreateLayer(3, 3)
	layer_2 := CreateLayer(3, 3)

	layer_1.Forward(X)
	fmt.Println(mat.Formatted(layer_1.Output))

	if layer_1 == nil || layer_2 == nil {
		t.Errorf("error: layer_1 & layer_2 are nil!")
	}
}
