package layer

import "testing"

func TestLayerCreation(t *testing.T) {
	layer_1 := CreateLayer(2, 3)
	layer_2 := CreateLayer(3, 3)

	if layer_1 == nil || layer_2 == nil {
		t.Errorf("error: layer_1 & layer_2 are nil!")
	}
}
