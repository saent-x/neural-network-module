package layer

type LayerNavigation struct {
	Prev interface{}
	Next interface{}
}

func (layerNavigation *LayerNavigation) GetPreviousLayer() interface{} {
	return layerNavigation.Prev
}

func (layerNavigation *LayerNavigation) GetNextLayer() interface{} {
	return layerNavigation.Next
}

func (layerNavigation *LayerNavigation) SetPreviousLayer(prev interface{}) {
	layerNavigation.Prev = prev
}

func (layerNavigation *LayerNavigation) SetNextLayer(next interface{}) {
	layerNavigation.Next = next
}
