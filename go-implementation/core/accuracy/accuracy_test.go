package accuracy

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestAccuracyFunction_1(t *testing.T) {
	softmax_outputs := mat.NewDense(3, 3, []float64{0.7, 0.2, 0.1, 0.5, 0.1, 0.4, 0.02, 0.9, 0.08})
	class_targets := mat.NewDense(1, 3, []float64{0, 1, 1})

	accuracy := Calculate(softmax_outputs, class_targets)

	got := accuracy
	wanted := 0.6666666666666666

	if got != wanted {
		t.Error("want", wanted, "got", got)
	}

	fmt.Println("got: ", got)
	fmt.Println("wanted: ", wanted)

}
